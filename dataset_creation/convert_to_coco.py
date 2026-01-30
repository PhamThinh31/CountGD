import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def countgd_odvg_to_coco(jsonl_path, images_dir, output_json_path, category_mapping=None):
    """
    Convert CountGD ODVG JSONL format to COCO JSON format

    CountGD ODVG format:
    {
        "filename": "image.jpg",
        "height": 384,
        "width": 512,
        "detection": {
            "instances": [
                {"bbox": [cx-1, cy-1, cx+1, cy+1], "label": 0, "category": "object_name"}
            ]
        },
        "exemplars": [[x1, y1, x2, y2], ...]
    }

    COCO format:
    {
        "info": {...},
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }
    """

    entries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))

    # Apply category mapping if provided
    if category_mapping:
        for entry in entries:
            for inst in entry['detection']['instances']:
                old_category = inst['category']
                if old_category in category_mapping:
                    inst['category'] = category_mapping[old_category]

    all_categories = set()
    for entry in entries:
        for inst in entry['detection']['instances']:
            all_categories.add(inst['category'])

    # Category IDs start from 0 to match cat_list indexing in inference
    category_to_id = {cat: idx for idx, cat in enumerate(sorted(all_categories))}

    coco_categories = [
        {
            "id": cat_id,
            "name": cat_name,
            "supercategory": "object"
        }
        for cat_name, cat_id in sorted(category_to_id.items(), key=lambda x: x[1])
    ]

    coco_images = []
    coco_annotations = []

    annotation_id = 1
    img_id = 1
    skipped_count = 0

    for entry in entries:
        # Skip images with no instances
        if not entry.get('detection') or not entry['detection'].get('instances'):
            print(f"⚠️  Skipping {entry['filename']} - no detection instances")
            skipped_count += 1
            continue

        if len(entry['detection']['instances']) == 0:
            print(f"⚠️  Skipping {entry['filename']} - empty instances list")
            skipped_count += 1
            continue

        exemplars = entry.get('exemplars', [])

        if len(exemplars) < 3:
            anns_for_exemplars = []
            for inst in entry['detection']['instances']:
                bbox = inst['bbox']
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    anns_for_exemplars.append({'bbox': [x1, y1, x2-x1, y2-y1]})

            if len(anns_for_exemplars) == 0:
                print(f"⚠️  Skipping {entry['filename']} - no valid annotations")
                skipped_count += 1
                continue

            from manual_coco_converter import select_exemplars
            exemplars = select_exemplars(anns_for_exemplars, num_exemplars=3)

        # Validate that we have exactly 3 exemplars
        if len(exemplars) != 3:
            print(f"⚠️  Skipping {entry['filename']} - could not generate 3 exemplars (got {len(exemplars)})")
            skipped_count += 1
            continue

        coco_images.append({
            "id": img_id,
            "file_name": entry['filename'],
            "height": entry['height'],
            "width": entry['width'],
            "exemplars": exemplars
        })

        # Add point annotations for all instances (area=4)
        for inst in entry['detection']['instances']:
            bbox = inst['bbox']

            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                x = cx
                y = cy
                w = 2
                h = 2
            else:
                continue

            area = 4

            coco_annotations.append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": category_to_id[inst['category']],
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0
            })

            annotation_id += 1

        # Add 3 exemplar annotations (area != 4) - these are the visual reference boxes
        for exemplar_bbox in exemplars:
            x1, y1, x2, y2 = exemplar_bbox
            w = x2 - x1
            h = y2 - y1
            area = w * h

            # Get category from first instance (all instances should be same category in CountGD)
            category_id = category_to_id[entry['detection']['instances'][0]['category']]

            coco_annotations.append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "area": area,  # Real area, not 4
                "iscrowd": 0
            })

            annotation_id += 1

        img_id += 1

    coco_data = {
        "info": {
            "description": "Converted from CountGD ODVG format",
            "date_created": datetime.now().isoformat(),
            "version": "1.0"
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"✅ Created COCO format: {output_json_path}")
    print(f"   Images: {len(coco_images)}")
    if skipped_count > 0:
        print(f"   Skipped: {skipped_count} images (no valid exemplars)")
    print(f"   Annotations: {len(coco_annotations)}")
    print(f"   Categories: {len(coco_categories)}")

    for cat in coco_categories:
        cat_anns = [a for a in coco_annotations if a['category_id'] == cat['id']]
        print(f"     {cat['name']:30s}: {len(cat_anns)} instances")

    return coco_data


def update_dataset_config(config_path, val_coco_path, test_coco_path, images_root):
    """Update dataset config to use COCO format for val/test"""

    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'val' in config and val_coco_path:
        config['val'] = [
            {
                "root": str(images_root),
                "anno": str(val_coco_path),
                "label_map": None,
                "dataset_mode": "coco"
            }
        ]

    if 'test' in config and test_coco_path:
        config['test'] = [
            {
                "root": str(images_root),
                "anno": str(test_coco_path),
                "label_map": None,
                "dataset_mode": "coco"
            }
        ]

    new_config_path = config_path.parent / "dataset_config_mixed.json"
    with open(new_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Created mixed config: {new_config_path}")
    print(f"   Train: ODVG format")
    print(f"   Val: COCO format")
    if test_coco_path:
        print(f"   Test: COCO format")

    return new_config_path


def main():
    parser = argparse.ArgumentParser("Convert CountGD ODVG to COCO format")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--category_mapping', type=str, help='Optional category mapping JSON')
    parser.add_argument('--convert_val', action='store_true', default=True, help='Convert val set')
    parser.add_argument('--convert_test', action='store_true', default=True, help='Convert test set')
    parser.add_argument('--convert_train', action='store_true', help='Also convert train set')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"

    category_mapping = None
    if args.category_mapping:
        with open(args.category_mapping, 'r') as f:
            category_mapping = json.load(f)

    print("="*80)
    print("CONVERTING TO COCO FORMAT")
    print("="*80)

    val_coco_path = None
    test_coco_path = None
    train_coco_path = None

    if args.convert_train:
        train_jsonl = data_dir / "train.jsonl"
        if train_jsonl.exists():
            print("\nConverting train set...")
            train_coco_path = data_dir / "train_coco.json"
            countgd_odvg_to_coco(train_jsonl, images_dir, train_coco_path, category_mapping)

    if args.convert_val:
        val_jsonl = data_dir / "val.jsonl"
        if val_jsonl.exists():
            print("\nConverting val set...")
            val_coco_path = data_dir / "val_coco.json"
            countgd_odvg_to_coco(val_jsonl, images_dir, val_coco_path, category_mapping)
        else:
            print("\n⚠️  val.jsonl not found, skipping")

    if args.convert_test:
        test_jsonl = data_dir / "test.jsonl"
        if test_jsonl.exists():
            print("\nConverting test set...")
            test_coco_path = data_dir / "test_coco.json"
            countgd_odvg_to_coco(test_jsonl, images_dir, test_coco_path, category_mapping)
        else:
            print("\n⚠️  test.jsonl not found, skipping")

    config_path = data_dir / "dataset_config.json"
    if config_path.exists():
        print("\nUpdating dataset config...")
        new_config = update_dataset_config(
            config_path,
            val_coco_path.absolute() if val_coco_path else None,
            test_coco_path.absolute() if test_coco_path else None,
            images_dir.absolute()
        )

    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print("\nCreated files:")
    if train_coco_path:
        print(f"  - {train_coco_path}")
    if val_coco_path:
        print(f"  - {val_coco_path}")
    if test_coco_path:
        print(f"  - {test_coco_path}")
    if config_path.exists():
        print(f"  - {data_dir / 'dataset_config_mixed.json'}")

    print("\nUse the mixed config for training:")
    print(f"  --datasets {data_dir / 'dataset_config_mixed.json'}")


if __name__ == '__main__':
    main()
