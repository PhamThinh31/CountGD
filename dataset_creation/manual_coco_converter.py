import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import shutil


def select_exemplars(annotations, num_exemplars=3, strategy='spatial'):
    """
    Select exemplar bounding boxes from annotations
    ALWAYS returns exactly num_exemplars boxes (duplicates if needed)

    Strategies:
    - spatial: select from different spatial regions
    - size: select boxes with different sizes
    - random: random selection
    """
    if len(annotations) == 0:
        return [[0, 0, 1, 1]] * num_exemplars

    if len(annotations) < num_exemplars:
        exemplars = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            exemplars.append([int(x), int(y), int(x + w), int(y + h)])

        while len(exemplars) < num_exemplars:
            exemplars.append(exemplars[0])

        return exemplars

    if strategy == 'spatial':
        sorted_anns = sorted(annotations, key=lambda x: x['bbox'][0])
        step = len(sorted_anns) // num_exemplars
        selected_indices = [i * step for i in range(num_exemplars)]

    elif strategy == 'size':
        sorted_anns = sorted(annotations, key=lambda x: x['bbox'][2] * ann['bbox'][3], reverse=True)
        step = len(sorted_anns) // num_exemplars
        selected_indices = [i * step for i in range(num_exemplars)]

    elif strategy == 'random':
        import random
        sorted_anns = annotations
        selected_indices = random.sample(range(len(annotations)), num_exemplars)

    else:
        sorted_anns = sorted(annotations, key=lambda x: x['bbox'][0])
        step = len(sorted_anns) // num_exemplars
        selected_indices = [i * step for i in range(num_exemplars)]

    exemplars = []
    for idx in selected_indices:
        bbox = sorted_anns[idx]['bbox']
        x, y, w, h = bbox
        exemplars.append([int(x), int(y), int(x + w), int(y + h)])

    return exemplars


def coco_to_countgd_manual(
    coco_json_path,
    output_jsonl_path,
    category_filter=None,
    min_objects=2,
    num_exemplars=3,
    point_annotation=True
):
    """
    Convert COCO format to CountGD ODVG JSONL

    Args:
        coco_json_path: Path to COCO JSON
        output_jsonl_path: Output JSONL path
        category_filter: List of category names to include (None = all)
        min_objects: Minimum number of objects per image
        num_exemplars: Number of exemplar boxes per image
        point_annotation: Use point annotations (center point) instead of boxes
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    print(f"Loaded COCO data: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    if category_filter:
        name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
        allowed_category_ids = set()
        for cat_name in category_filter:
            if cat_name in name_to_id:
                allowed_category_ids.add(name_to_id[cat_name])
            else:
                print(f"Warning: category '{cat_name}' not found in COCO data")
    else:
        allowed_category_ids = set(category_id_to_name.keys())

    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['category_id'] in allowed_category_ids:
            image_annotations[ann['image_id']].append(ann)

    filtered_images = {
        img_id: anns
        for img_id, anns in image_annotations.items()
        if len(anns) >= min_objects
    }

    print(f"Filtered to {len(filtered_images)} images with >= {min_objects} objects")

    output_entries = []

    for img_id, anns in tqdm(filtered_images.items(), desc="Converting"):
        img_info = image_id_to_info[img_id]

        instances = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            cx = x + w / 2
            cy = y + h / 2

            category_name = category_id_to_name[ann['category_id']]

            if point_annotation:
                bbox = [cx - 1, cy - 1, cx + 1, cy + 1]
            else:
                bbox = [x, y, x + w, y + h]

            instances.append({
                "bbox": bbox,
                "label": 0,
                "category": category_name
            })

        exemplars = select_exemplars(anns, num_exemplars=num_exemplars)

        entry = {
            "filename": img_info['file_name'],
            "height": img_info['height'],
            "width": img_info['width'],
            "detection": {
                "instances": instances
            },
            "exemplars": exemplars
        }

        output_entries.append(entry)

    with open(output_jsonl_path, 'w') as f:
        for entry in output_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(output_entries)} entries to {output_jsonl_path}")

    stats = analyze_converted_data(output_entries)

    return output_entries, stats


def analyze_converted_data(entries):
    """Analyze converted JSONL data"""
    stats = {
        'total_images': len(entries),
        'total_objects': 0,
        'objects_per_image': [],
        'categories': defaultdict(int),
        'image_sizes': [],
    }

    for entry in entries:
        num_objects = len(entry['detection']['instances'])
        stats['total_objects'] += num_objects
        stats['objects_per_image'].append(num_objects)
        stats['image_sizes'].append((entry['width'], entry['height']))

        for inst in entry['detection']['instances']:
            stats['categories'][inst['category']] += 1

    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Total images: {stats['total_images']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Avg objects/image: {np.mean(stats['objects_per_image']):.2f} ± {np.std(stats['objects_per_image']):.2f}")
    print(f"Min/Max objects: {np.min(stats['objects_per_image'])} / {np.max(stats['objects_per_image'])}")

    print(f"\nCategory distribution:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_objects'] * 100
        print(f"  {cat:30s}: {count:6d} ({pct:5.1f}%)")

    widths = [s[0] for s in stats['image_sizes']]
    heights = [s[1] for s in stats['image_sizes']]
    print(f"\nImage dimensions:")
    print(f"  Width:  {np.min(widths):4d} - {np.max(widths):4d} (avg: {np.mean(widths):.1f})")
    print(f"  Height: {np.min(heights):4d} - {np.max(heights):4d} (avg: {np.mean(heights):.1f})")
    print("="*80)

    return stats


def create_train_val_split(entries, split_ratio=0.8, output_dir='.'):
    """Split data into train/val sets"""
    import random
    random.shuffle(entries)

    split_idx = int(len(entries) * split_ratio)
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]

    output_dir = Path(output_dir)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with open(train_path, 'w') as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + '\n')

    with open(val_path, 'w') as f:
        for entry in val_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\nTrain: {len(train_entries)} images -> {train_path}")
    print(f"Val:   {len(val_entries)} images -> {val_path}")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser("Manual COCO to CountGD converter")
    parser.add_argument('--coco_json', type=str, required=True, help='Path to COCO JSON')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Output JSONL path')
    parser.add_argument('--category_filter', nargs='+', help='Filter by category names')
    parser.add_argument('--min_objects', type=int, default=2, help='Min objects per image')
    parser.add_argument('--num_exemplars', type=int, default=3, help='Number of exemplars')
    parser.add_argument('--point_annotation', action='store_true', help='Use point annotations')
    parser.add_argument('--split', action='store_true', help='Create train/val split')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train split ratio')

    args = parser.parse_args()

    entries, stats = coco_to_countgd_manual(
        args.coco_json,
        args.output_jsonl,
        category_filter=args.category_filter,
        min_objects=args.min_objects,
        num_exemplars=args.num_exemplars,
        point_annotation=args.point_annotation
    )

    if args.split:
        output_dir = Path(args.output_jsonl).parent
        create_train_val_split(entries, args.split_ratio, output_dir)


if __name__ == '__main__':
    main()
