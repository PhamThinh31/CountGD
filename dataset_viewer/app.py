"""
Flask app to visualize CountGD dataset statistics.

Usage:
    python app.py --config path/to/dataset_config.json [--port 5041]

    # With raw YOLO debugging:
    python app.py --config path/to/dataset_config.json --raw_yolo_dir /path/to/multi_classes_dataset
"""

import argparse
import io
import json
import math
import yaml
from pathlib import Path
from collections import defaultdict

from flask import Flask, render_template, request, send_file, abort, jsonify
from PIL import Image, ImageDraw

from dataset_parser import load_dataset_config

app = Flask(__name__)

# Global state filled at startup
DATASETS = {}       # {split_name: [parsed_dataset, ...]}
CONFIG_PATH = ""
RAW_YOLO_DIR = None  # Optional: path to original YOLO dataset for debugging
RAW_YOLO_INDEX = {}  # original_filename (without d{idx}_ prefix) -> {label_path, images_dir, dataset_folder}
YOLO_DATASET_FOLDERS = []  # sorted list of dataset folder names, index = dataset_idx


def get_dataset_key(split, idx):
    return f"{split}_{idx}"


def get_dataset(split, idx):
    idx = int(idx)
    entries = DATASETS.get(split, [])
    if idx < 0 or idx >= len(entries):
        return None
    return entries[idx]


# ─── Raw YOLO index builder ──────────────────────────────────────────────────

def _build_yolo_index(raw_yolo_dir):
    """Build an index from converted filenames back to raw YOLO label files.

    Converted filenames are d{idx}_{original_name}.
    idx matches the sorted order of subfolders in raw_yolo_dir.
    """
    global RAW_YOLO_INDEX, YOLO_DATASET_FOLDERS

    raw_yolo_dir = Path(raw_yolo_dir)
    folders = sorted([
        d for d in raw_yolo_dir.iterdir()
        if d.is_dir() and (d / "data.yaml").exists()
    ])
    YOLO_DATASET_FOLDERS = [f.name for f in folders]

    count = 0
    for dataset_idx, folder in enumerate(folders):
        for split_name in ["train", "valid", "test"]:
            images_dir = folder / split_name / "images"
            labels_dir = folder / split_name / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                continue
            for img_path in images_dir.iterdir():
                if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                label_path = labels_dir / (img_path.stem + ".txt")
                if not label_path.exists():
                    continue
                # key = the converted filename: d{idx}_{original}
                converted_name = f"d{dataset_idx}_{img_path.name}"
                RAW_YOLO_INDEX[converted_name] = {
                    "label_path": str(label_path),
                    "image_path": str(img_path),
                    "dataset_folder": folder.name,
                    "split": split_name,
                }
                count += 1

    print(f"  Raw YOLO index: {count} images from {len(folders)} datasets")


def _read_yolo_bboxes(label_path, img_w, img_h):
    """Read a YOLO label file and return denormalized bboxes.

    Returns list of [x1, y1, x2, y2] in pixel coords.
    """
    bboxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx_norm = float(parts[1])
            cy_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            cx = cx_norm * img_w
            cy = cy_norm * img_h
            w = w_norm * img_w
            h = h_norm * img_h

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            bboxes.append([x1, y1, x2, y2])
    return bboxes


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
    return render_template(
        "dashboard.html",
        datasets=summary,
        config_path=CONFIG_PATH,
        has_raw_yolo=bool(RAW_YOLO_DIR),
    )


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
        has_raw_yolo=bool(RAW_YOLO_DIR),
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
        has_yolo = img_info["file_name"] in RAW_YOLO_INDEX
        images.append({
            "id": img_id,
            "file_name": img_info["file_name"],
            "width": img_info["width"],
            "height": img_info["height"],
            "count": count,
            "has_yolo": has_yolo,
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
        has_raw_yolo=bool(RAW_YOLO_DIR),
    )


@app.route("/image/<split>/<int:idx>/<int:img_id>")
def serve_image(split, idx, img_id):
    """Serve an image with annotations drawn on it.

    Query params:
        draw=0|1       - draw converted annotations (default 1)
        draw_yolo=0|1  - draw raw YOLO bboxes (default 0)
        cat=<name>     - filter annotations to this category
    """
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
    draw_yolo = request.args.get("draw_yolo", "0") == "1"
    cat_filter = request.args.get("cat", None)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # ── Layer 1: Raw YOLO bounding boxes (green) ──
    if draw_yolo:
        yolo_entry = RAW_YOLO_INDEX.get(img_info["file_name"])
        if yolo_entry:
            bboxes = _read_yolo_bboxes(
                yolo_entry["label_path"], img.width, img.height
            )
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="#22c55e", width=2)

    if draw_annos:
        # ── Layer 2: Exemplar boxes (blue) ──
        for ex in img_info.get("exemplars", []):
            bbox = ex["bbox"]
            if ds["format"] == "coco":
                x, y, w, h = bbox
                coords = [x, y, x + w, y + h]
            else:
                coords = bbox
            draw.rectangle(coords, outline="#3b82f6", width=3)

        # ── Layer 3: Converted point annotations (red dots) ──
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


# ─── Debug info API ───────────────────────────────────────────────────────────

@app.route("/api/yolo_info/<split>/<int:idx>/<int:img_id>")
def yolo_info(split, idx, img_id):
    """Return raw YOLO debug info for an image."""
    ds = get_dataset(split, idx)
    if ds is None:
        abort(404)

    img_info = ds["images"].get(img_id)
    if img_info is None:
        abort(404)

    filename = img_info["file_name"]
    yolo_entry = RAW_YOLO_INDEX.get(filename)

    if not yolo_entry:
        return jsonify({
            "found": False,
            "filename": filename,
            "message": "No raw YOLO data found for this image",
        })

    bboxes = _read_yolo_bboxes(
        yolo_entry["label_path"], img_info["width"], img_info["height"]
    )

    converted_count = len(img_info["annotations"])

    return jsonify({
        "found": True,
        "filename": filename,
        "dataset_folder": yolo_entry["dataset_folder"],
        "split": yolo_entry["split"],
        "label_path": yolo_entry["label_path"],
        "raw_yolo_count": len(bboxes),
        "converted_count": converted_count,
        "match": len(bboxes) == converted_count,
        "raw_bboxes": bboxes,
    })


@app.route("/dataset/<split>/<int:idx>/category/<cat_name>/stats")
def category_stats_json(split, idx, cat_name):
    """Return raw JSON stats for a category."""
    ds = get_dataset(split, idx)
    if ds is None:
        abort(404)
    cat = ds["categories"].get(cat_name)
    if cat is None:
        abort(404)

    return jsonify({
        "name": cat["name"],
        "image_count": cat["image_count"],
        "annotation_count": cat["annotation_count"],
        "avg_count_per_image": cat["avg_count_per_image"],
        "min_count": cat["min_count"],
        "max_count": cat["max_count"],
    })


# ─── Startup ─────────────────────────────────────────────────────────────────

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


def init_app(config_path, raw_yolo_dir=None):
    global DATASETS, CONFIG_PATH, RAW_YOLO_DIR
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

    if raw_yolo_dir:
        RAW_YOLO_DIR = str(raw_yolo_dir)
        print(f"\nBuilding raw YOLO index from: {raw_yolo_dir}")
        _build_yolo_index(raw_yolo_dir)

    print("Ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CountGD Dataset Viewer")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config JSON")
    parser.add_argument("--raw_yolo_dir", type=str, default=None,
                        help="Path to original YOLO dataset folder for debug comparison")
    parser.add_argument("--port", type=int, default=5041, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    init_app(args.config, args.raw_yolo_dir)
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
