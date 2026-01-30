"""
Flask app to visualize CountGD dataset statistics.

Usage:
    python app.py --config path/to/dataset_config.json [--port 5000]
"""

import argparse
import io
import math
from pathlib import Path

from flask import Flask, render_template, request, send_file, abort
from PIL import Image, ImageDraw

from dataset_parser import load_dataset_config

app = Flask(__name__)

# Global state filled at startup
DATASETS = {}       # {split_name: [parsed_dataset, ...]}
CONFIG_PATH = ""


def get_dataset_key(split, idx):
    return f"{split}_{idx}"


def get_dataset(split, idx):
    idx = int(idx)
    entries = DATASETS.get(split, [])
    if idx < 0 or idx >= len(entries):
        return None
    return entries[idx]


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def dashboard():
    summary = []
    for split_name, entries in DATASETS.items():
        for idx, ds in enumerate(entries):
            summary.append({
                "key": get_dataset_key(split_name, idx),
                "split": split_name,
                "idx": idx,
                "format": ds["format"],
                "anno": ds["anno"],
                "root": ds["root"],
                "total_images": ds["total_images"],
                "total_annotations": ds["total_annotations"],
                "total_exemplars": ds["total_exemplars"],
                "total_categories": ds["total_categories"],
                "top_categories": sorted(
                    ds["categories"].values(),
                    key=lambda c: c["annotation_count"],
                    reverse=True,
                )[:5],
            })
    return render_template("dashboard.html", datasets=summary, config_path=CONFIG_PATH)


@app.route("/dataset/<split>/<int:idx>")
def dataset_detail(split, idx):
    ds = get_dataset(split, idx)
    if ds is None:
        abort(404)

    cats = sorted(ds["categories"].values(), key=lambda c: c["annotation_count"], reverse=True)

    # Distribution histogram data: count_per_image across all images
    all_counts = []
    for cat in cats:
        s = ds["categories"][cat["name"]]
        all_counts.extend(s.get("counts_per_image", []) if "counts_per_image" in s else [])

    # Build histogram buckets
    if all_counts:
        max_count = max(all_counts)
        bucket_size = max(1, max_count // 20)
        hist = {}
        for c in all_counts:
            bucket = (c // bucket_size) * bucket_size
            hist[bucket] = hist.get(bucket, 0) + 1
        histogram = sorted(hist.items())
    else:
        histogram = []

    return render_template(
        "dataset_detail.html",
        ds=ds,
        split=split,
        idx=idx,
        categories=cats,
        histogram=histogram,
    )


@app.route("/dataset/<split>/<int:idx>/category/<cat_name>")
def category_detail(split, idx, cat_name):
    ds = get_dataset(split, idx)
    if ds is None:
        abort(404)

    cat = ds["categories"].get(cat_name)
    if cat is None:
        abort(404)

    page = request.args.get("page", 1, type=int)
    per_page = 30
    image_ids = cat["image_ids"]
    total_pages = max(1, math.ceil(len(image_ids) / per_page))
    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    end = start + per_page
    page_ids = image_ids[start:end]

    images = []
    for img_id in page_ids:
        img_info = ds["images"].get(img_id)
        if img_info is None:
            continue
        count = sum(1 for a in img_info["annotations"] if a["category_name"] == cat_name)
        images.append({
            "id": img_id,
            "file_name": img_info["file_name"],
            "width": img_info["width"],
            "height": img_info["height"],
            "count": count,
        })

    # Counts distribution for this category
    counts_per_image = ds["categories"][cat_name].get("counts_per_image", [])
    if counts_per_image:
        max_c = max(counts_per_image)
        bucket_size = max(1, max_c // 15)
        hist = {}
        for c in counts_per_image:
            bucket = (c // bucket_size) * bucket_size
            hist[bucket] = hist.get(bucket, 0) + 1
        histogram = sorted(hist.items())
    else:
        histogram = []

    # Reparse the internal stats to pass counts_per_image
    cat_with_stats = dict(cat)
    cat_with_stats["counts_per_image"] = counts_per_image

    return render_template(
        "category_detail.html",
        ds=ds,
        split=split,
        idx=idx,
        cat=cat_with_stats,
        images=images,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        total_images=len(image_ids),
        histogram=histogram,
    )


@app.route("/image/<split>/<int:idx>/<int:img_id>")
def serve_image(split, idx, img_id):
    """Serve an image with annotations drawn on it."""
    ds = get_dataset(split, idx)
    if ds is None:
        abort(404)

    img_info = ds["images"].get(img_id)
    if img_info is None:
        abort(404)

    root = Path(ds["root"])
    img_path = root / img_info["file_name"]
    if not img_path.exists():
        abort(404)

    draw_annos = request.args.get("draw", "1") == "1"
    cat_filter = request.args.get("cat", None)

    img = Image.open(img_path).convert("RGB")

    if draw_annos:
        draw = ImageDraw.Draw(img)

        # Draw exemplars in blue
        for ex in img_info.get("exemplars", []):
            bbox = ex["bbox"]
            if ds["format"] == "coco":
                # COCO: [x, y, w, h]
                x, y, w, h = bbox
                coords = [x, y, x + w, y + h]
            else:
                # ODVG: [x1, y1, x2, y2]
                coords = bbox
            draw.rectangle(coords, outline="#3b82f6", width=3)

        # Draw point annotations
        for ann in img_info["annotations"]:
            if cat_filter and ann["category_name"] != cat_filter:
                continue
            bbox = ann["bbox"]
            if ds["format"] == "coco":
                x, y, w, h = bbox
                cx, cy = x + w / 2, y + h / 2
            else:
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            r = 6
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#ef4444", outline="#ffffff", width=1)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/image_raw/<split>/<int:idx>/<int:img_id>")
def serve_image_raw(split, idx, img_id):
    """Serve the raw image without annotations."""
    ds = get_dataset(split, idx)
    if ds is None:
        abort(404)

    img_info = ds["images"].get(img_id)
    if img_info is None:
        abort(404)

    root = Path(ds["root"])
    img_path = root / img_info["file_name"]
    if not img_path.exists():
        abort(404)

    return send_file(str(img_path))


# ─── Category stats API ──────────────────────────────────────────────────────

@app.route("/dataset/<split>/<int:idx>/category/<cat_name>/stats")
def category_stats_json(split, idx, cat_name):
    """Return raw JSON stats for a category (for any frontend charting)."""
    import json as json_module
    from flask import jsonify

    ds = get_dataset(split, idx)
    if ds is None:
        abort(404)
    cat = ds["categories"].get(cat_name)
    if cat is None:
        abort(404)

    # Safe copy without image_ids (can be huge)
    return jsonify({
        "name": cat["name"],
        "image_count": cat["image_count"],
        "annotation_count": cat["annotation_count"],
        "avg_count_per_image": cat["avg_count_per_image"],
        "min_count": cat["min_count"],
        "max_count": cat["max_count"],
    })


# ─── Startup ─────────────────────────────────────────────────────────────────

# Rebuild the internal per-category counts_per_image list in the COCO parser
# so it's available in dataset_parser (it already stores it).
# We need to also store it for COCO format since parse_coco uses a defaultdict.

def _rebuild_counts_per_image(ds):
    """Rebuild counts_per_image for each category from image data."""
    for cat_name, cat_info in ds["categories"].items():
        if "counts_per_image" in cat_info:
            continue
        counts = []
        for img_id in cat_info.get("image_ids", []):
            img = ds["images"].get(img_id)
            if img is None:
                continue
            c = sum(1 for a in img["annotations"] if a["category_name"] == cat_name)
            counts.append(c)
        cat_info["counts_per_image"] = counts


def init_app(config_path):
    global DATASETS, CONFIG_PATH
    CONFIG_PATH = str(config_path)
    print(f"Loading datasets from: {config_path}")
    DATASETS = load_dataset_config(config_path)

    for split, entries in DATASETS.items():
        for ds in entries:
            _rebuild_counts_per_image(ds)
            print(f"  [{split}] {ds['format'].upper()} | "
                  f"{ds['total_images']} images | "
                  f"{ds['total_annotations']} annotations | "
                  f"{ds['total_categories']} categories")

    print("Ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CountGD Dataset Viewer")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config JSON")
    parser.add_argument("--port", type=int, default=5041, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    init_app(args.config)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
