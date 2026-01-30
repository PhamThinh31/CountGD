"""
Parse COCO and ODVG dataset formats into unified statistics structures.
"""

import json
from pathlib import Path
from collections import defaultdict


def parse_coco(anno_path, root_dir):
    """Parse a COCO JSON annotation file.

    Returns:
        dict with keys:
            format: "coco"
            root: str
            anno: str
            total_images: int
            total_annotations: int  (point annotations only, area==4)
            total_exemplars: int
            categories: dict of cat_name -> {
                id, name, image_count, annotation_count,
                avg_count_per_image, min_count, max_count,
                image_ids: [int]
            }
            images: dict of image_id -> {
                id, file_name, width, height,
                annotations: [{bbox, category_id, category_name, area}],
                exemplars: [{bbox, category_id, category_name, area}],
                categories: set of category names
            }
    """
    with open(anno_path, "r") as f:
        data = json.load(f)

    cat_id_to_name = {}
    for cat in data.get("categories", []):
        cat_id_to_name[cat["id"]] = cat["name"]

    images_by_id = {}
    for img in data.get("images", []):
        images_by_id[img["id"]] = {
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "annotations": [],
            "exemplars": [],
            "categories": set(),
        }

    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in images_by_id:
            continue
        cat_name = cat_id_to_name.get(ann["category_id"], f"unknown_{ann['category_id']}")
        entry = {
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
            "category_name": cat_name,
            "area": ann.get("area", 0),
        }
        if ann.get("area", 0) == 4:
            images_by_id[img_id]["annotations"].append(entry)
            images_by_id[img_id]["categories"].add(cat_name)
        else:
            images_by_id[img_id]["exemplars"].append(entry)

    # Build per-category stats
    cat_stats = defaultdict(lambda: {
        "image_count": 0,
        "annotation_count": 0,
        "counts_per_image": [],
        "image_ids": [],
    })

    for img_id, img_info in images_by_id.items():
        per_cat_count = defaultdict(int)
        for ann in img_info["annotations"]:
            per_cat_count[ann["category_name"]] += 1
        for cat_name, count in per_cat_count.items():
            cat_stats[cat_name]["image_count"] += 1
            cat_stats[cat_name]["annotation_count"] += count
            cat_stats[cat_name]["counts_per_image"].append(count)
            cat_stats[cat_name]["image_ids"].append(img_id)

    categories = {}
    for cat in data.get("categories", []):
        name = cat["name"]
        s = cat_stats[name]
        counts = s["counts_per_image"] if s["counts_per_image"] else [0]
        categories[name] = {
            "id": cat["id"],
            "name": name,
            "image_count": s["image_count"],
            "annotation_count": s["annotation_count"],
            "avg_count_per_image": round(sum(counts) / len(counts), 1) if counts else 0,
            "min_count": min(counts) if counts else 0,
            "max_count": max(counts) if counts else 0,
            "image_ids": s["image_ids"],
        }

    total_anns = sum(1 for img in images_by_id.values() for _ in img["annotations"])
    total_exemplars = sum(1 for img in images_by_id.values() for _ in img["exemplars"])

    return {
        "format": "coco",
        "root": str(root_dir),
        "anno": str(anno_path),
        "total_images": len(images_by_id),
        "total_annotations": total_anns,
        "total_exemplars": total_exemplars,
        "total_categories": len(categories),
        "categories": categories,
        "images": images_by_id,
    }


def parse_odvg(anno_path, root_dir, label_map_path=None):
    """Parse an ODVG JSONL annotation file.

    Returns same unified structure as parse_coco.
    """
    label_map = {}
    if label_map_path and Path(label_map_path).exists():
        with open(label_map_path, "r") as f:
            label_map = json.load(f)

    images_by_id = {}
    cat_stats = defaultdict(lambda: {
        "image_count": 0,
        "annotation_count": 0,
        "counts_per_image": [],
        "image_ids": [],
    })
    all_categories = {}

    with open(anno_path, "r") as f:
        for img_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            instances = entry.get("detection", {}).get("instances", [])
            exemplars_raw = entry.get("exemplars", [])

            annotations = []
            exemplars = []
            categories_in_img = set()
            per_cat_count = defaultdict(int)

            for inst in instances:
                bbox = inst["bbox"]
                cat_name = inst.get("category", label_map.get(str(inst.get("label", 0)), f"class_{inst.get('label', 0)}"))
                label_id = inst.get("label", 0)

                annotations.append({
                    "bbox": bbox,
                    "category_id": label_id,
                    "category_name": cat_name,
                    "area": 4,
                })
                categories_in_img.add(cat_name)
                per_cat_count[cat_name] += 1

                if cat_name not in all_categories:
                    all_categories[cat_name] = {"id": label_id, "name": cat_name}

            for ex_bbox in exemplars_raw:
                exemplars.append({
                    "bbox": ex_bbox,
                    "category_id": 0,
                    "category_name": "",
                    "area": abs((ex_bbox[2] - ex_bbox[0]) * (ex_bbox[3] - ex_bbox[1])) if len(ex_bbox) == 4 else 0,
                })

            images_by_id[img_id] = {
                "id": img_id,
                "file_name": entry["filename"],
                "width": entry.get("width", 0),
                "height": entry.get("height", 0),
                "annotations": annotations,
                "exemplars": exemplars,
                "categories": categories_in_img,
            }

            for cat_name, count in per_cat_count.items():
                cat_stats[cat_name]["image_count"] += 1
                cat_stats[cat_name]["annotation_count"] += count
                cat_stats[cat_name]["counts_per_image"].append(count)
                cat_stats[cat_name]["image_ids"].append(img_id)

    categories = {}
    for cat_name, cat_info in all_categories.items():
        s = cat_stats[cat_name]
        counts = s["counts_per_image"] if s["counts_per_image"] else [0]
        categories[cat_name] = {
            "id": cat_info["id"],
            "name": cat_name,
            "image_count": s["image_count"],
            "annotation_count": s["annotation_count"],
            "avg_count_per_image": round(sum(counts) / len(counts), 1) if counts else 0,
            "min_count": min(counts) if counts else 0,
            "max_count": max(counts) if counts else 0,
            "image_ids": s["image_ids"],
        }

    total_anns = sum(c["annotation_count"] for c in categories.values())
    total_exemplars = sum(len(img["exemplars"]) for img in images_by_id.values())

    return {
        "format": "odvg",
        "root": str(root_dir),
        "anno": str(anno_path),
        "total_images": len(images_by_id),
        "total_annotations": total_anns,
        "total_exemplars": total_exemplars,
        "total_categories": len(categories),
        "categories": categories,
        "images": images_by_id,
    }


def load_dataset_config(config_path):
    """Load a dataset config JSON and parse all datasets.

    Args:
        config_path: Path to dataset_config.json

    Returns:
        dict of split_name -> list of parsed dataset dicts
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    result = {}

    for split_name, entries in config.items():
        result[split_name] = []
        for entry in entries:
            root_dir = entry["root"]
            anno_path = entry["anno"]
            label_map = entry.get("label_map")
            mode = entry.get("dataset_mode", "coco")

            # Resolve relative paths: try config dir first, then its parent
            # (configs often live in config/ but paths are relative to project root)
            base_dir = config_path.parent

            def resolve_path(p):
                if Path(p).is_absolute():
                    return p
                candidate = (base_dir / p).resolve()
                if candidate.exists():
                    return str(candidate)
                candidate2 = (base_dir.parent / p).resolve()
                if candidate2.exists():
                    return str(candidate2)
                return str(candidate)  # fallback to config dir

            root_dir = resolve_path(root_dir)
            anno_path = resolve_path(anno_path)
            if label_map:
                label_map = resolve_path(label_map)

            if mode == "coco":
                parsed = parse_coco(anno_path, root_dir)
            else:
                parsed = parse_odvg(anno_path, root_dir, label_map)

            parsed["split"] = split_name
            parsed["config_entry"] = entry
            result[split_name].append(parsed)

    return result
