import json
import argparse
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from manual_coco_converter import select_exemplars
from convert_to_coco import countgd_odvg_to_coco, update_dataset_config


def parse_data_yaml(yaml_path):
    """Read data.yaml and extract class names."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', []), data.get('nc', 0)


def discover_datasets(input_dir):
    """Scan input_dir for subdirectories containing data.yaml.
    Returns list of (folder_path, category_name) tuples and a name mapping dict.
    """
    input_dir = Path(input_dir)
    datasets = []
    name_mapping = {}

    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        yaml_path = subdir / 'data.yaml'
        if not yaml_path.exists():
            continue

        names, nc = parse_data_yaml(yaml_path)
        if not names:
            print(f"Skipping {subdir.name}: no class names in data.yaml")
            continue

        # Use the first class name as the category
        category_name = names[0]
        folder_name = subdir.name

        name_mapping[folder_name] = category_name
        datasets.append((subdir, category_name))
        print(f"  {folder_name} -> {category_name}")

    return datasets, name_mapping


def convert_yolo_split(split_dir, category_name, min_objects=2, num_exemplars=3):
    """Convert one YOLO split (train/valid/test) folder to ODVG entries.

    Args:
        split_dir: Path to e.g. dataset/train/ containing images/ and labels/
        category_name: Category string for all objects
        min_objects: Skip images with fewer objects
        num_exemplars: Number of exemplar boxes per image

    Returns:
        List of ODVG entry dicts, each with extra '_source_img_path' key
    """
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labels'

    if not images_dir.exists() or not labels_dir.exists():
        return []

    entries = []
    image_files = sorted(images_dir.glob('*'))

    for img_path in image_files:
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
            continue

        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue

        # Read image dimensions
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception:
            continue

        # Parse YOLO labels
        annotations = []  # for exemplar selection (COCO-style bbox)
        instances = []     # for ODVG point annotations

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # YOLO: class_id cx_norm cy_norm w_norm h_norm
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Denormalize to pixel coords
                cx = cx_norm * img_w
                cy = cy_norm * img_h
                w = w_norm * img_w
                h = h_norm * img_h

                x1 = cx - w / 2
                y1 = cy - h / 2

                # COCO-style bbox for exemplar selection
                annotations.append({'bbox': [x1, y1, w, h]})

                # Point annotation for detection
                instances.append({
                    "bbox": [cx - 1, cy - 1, cx + 1, cy + 1],
                    "label": 0,
                    "category": category_name
                })

        if len(instances) < min_objects:
            continue

        exemplars = select_exemplars(annotations, num_exemplars=num_exemplars)

        entry = {
            "filename": img_path.name,
            "height": img_h,
            "width": img_w,
            "detection": {"instances": instances},
            "exemplars": exemplars,
            "_source_img_path": str(img_path),
        }
        entries.append(entry)

    return entries


def merge_and_write(all_split_entries, output_dir, name_mapping):
    """Merge entries from all datasets per split, copy images, write outputs.

    Args:
        all_split_entries: dict of split_name -> list of entry dicts
        output_dir: Path to output directory
        name_mapping: dict of folder_name -> category_name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True, parents=True)

    split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
    written_splits = {}

    for orig_split, target_split in split_map.items():
        entries = all_split_entries.get(orig_split, [])
        if not entries:
            continue

        # Copy images and update filenames
        for entry in tqdm(entries, desc=f"Copying {target_split} images"):
            src_path = Path(entry.pop('_source_img_path'))
            # Use dataset_idx prefix to avoid filename collisions
            dataset_prefix = entry.pop('_dataset_idx', '0')
            new_name = f"d{dataset_prefix}_{entry['filename']}"
            dst_path = images_dir / new_name

            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

            entry['filename'] = new_name

        # Write JSONL
        jsonl_path = output_dir / f"{target_split}.jsonl"
        with open(jsonl_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        written_splits[target_split] = (jsonl_path, len(entries))
        print(f"  {target_split}: {len(entries)} images -> {jsonl_path}")

    # Label map from all categories across all splits
    all_categories = set()
    for entries in all_split_entries.values():
        for entry in entries:
            for inst in entry['detection']['instances']:
                all_categories.add(inst['category'])

    label_map = {str(i): cat for i, cat in enumerate(sorted(all_categories))}
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"  Label map ({len(label_map)} categories) -> {label_map_path}")

    # Name mapping
    mapping_path = output_dir / "name_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(name_mapping, f, indent=2)
    print(f"  Name mapping -> {mapping_path}")

    # Dataset config (ODVG for train, ODVG for val/test initially)
    dataset_config = {}
    if 'train' in written_splits:
        dataset_config["train"] = [{
            "root": str(images_dir.absolute()),
            "anno": str((output_dir / "train.jsonl").absolute()),
            "label_map": str(label_map_path.absolute()),
            "dataset_mode": "odvg"
        }]
    if 'val' in written_splits:
        dataset_config["val"] = [{
            "root": str(images_dir.absolute()),
            "anno": str((output_dir / "val.jsonl").absolute()),
            "label_map": None,
            "dataset_mode": "odvg"
        }]
    if 'test' in written_splits:
        dataset_config["test"] = [{
            "root": str(images_dir.absolute()),
            "anno": str((output_dir / "test.jsonl").absolute()),
            "label_map": None,
            "dataset_mode": "odvg"
        }]

    config_path = output_dir / "dataset_config.json"
    with open(config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)

    # Convert val/test to COCO format
    print("\nConverting val/test to COCO format...")
    val_coco_path = None
    test_coco_path = None

    if 'val' in written_splits:
        val_coco_path = output_dir / "val_coco.json"
        countgd_odvg_to_coco(
            output_dir / "val.jsonl", images_dir, val_coco_path
        )

    if 'test' in written_splits:
        test_coco_path = output_dir / "test_coco.json"
        countgd_odvg_to_coco(
            output_dir / "test.jsonl", images_dir, test_coco_path
        )

    # Mixed config (ODVG train, COCO val/test)
    if val_coco_path or test_coco_path:
        update_dataset_config(
            config_path,
            val_coco_path.absolute() if val_coco_path else None,
            test_coco_path.absolute() if test_coco_path else None,
            images_dir.absolute()
        )

    return config_path, written_splits


def main():
    parser = argparse.ArgumentParser(
        description="Convert downloaded Roboflow YOLO datasets to CountGD ODVG format"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to folder containing YOLO dataset subfolders')
    parser.add_argument('--output_dir', type=str, default='yolo_converted',
                        help='Output directory')
    parser.add_argument('--min_objects', type=int, default=2,
                        help='Minimum objects per image')
    parser.add_argument('--num_exemplars', type=int, default=3,
                        help='Number of exemplar boxes per image')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("YOLO -> CountGD ODVG Converter")
    print("=" * 80)

    # 1. Discover datasets
    print(f"\nScanning {input_dir} for datasets...")
    datasets, name_mapping = discover_datasets(input_dir)
    print(f"Found {len(datasets)} datasets\n")

    if not datasets:
        print("No datasets found. Make sure subfolders contain data.yaml.")
        return

    # 2. Convert each dataset, keeping original splits
    all_split_entries = defaultdict(list)

    for idx, (dataset_dir, category_name) in enumerate(datasets):
        print(f"\n[{idx+1}/{len(datasets)}] Converting: {dataset_dir.name} ({category_name})")

        for split_name in ['train', 'valid', 'test']:
            split_dir = dataset_dir / split_name
            if not split_dir.exists():
                continue

            entries = convert_yolo_split(
                split_dir, category_name,
                min_objects=args.min_objects,
                num_exemplars=args.num_exemplars
            )

            # Tag entries with dataset index for filename prefixing
            for entry in entries:
                entry['_dataset_idx'] = str(idx)

            all_split_entries[split_name].extend(entries)
            print(f"  {split_name}: {len(entries)} images")

    # 3. Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for split_name in ['train', 'valid', 'test']:
        count = len(all_split_entries.get(split_name, []))
        if count > 0:
            print(f"  {split_name}: {count} images")

    # 4. Merge and write
    print(f"\nWriting output to {output_dir}...")
    config_path, written_splits = merge_and_write(
        all_split_entries, output_dir, name_mapping
    )

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")
    print(f"Output: {output_dir}")
    print(f"Config: {config_path}")
    print(f"Mixed config: {output_dir / 'dataset_config_mixed.json'}")
    print(f"Name mapping: {output_dir / 'name_mapping.json'}")
    print(f"\nUse for training:")
    print(f"  --datasets {output_dir / 'dataset_config_mixed.json'}")


if __name__ == '__main__':
    main()
