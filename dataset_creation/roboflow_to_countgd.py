import os
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import subprocess
from tqdm import tqdm
import requests
from PIL import Image
import numpy as np


def extract_roboflow_details(url):
    """
    Parses a Roboflow Universe URL to extract workspace, project, and version.
    """
    clean_url = url.split("?")[0]
    parts = clean_url.replace("https://", "").replace("http://", "").split("/")

    if len(parts) < 3:
        return None, None, None

    workspace = parts[1]
    project = parts[2]

    version = None
    if "model" in parts:
        try:
            model_index = parts.index("model")
            if len(parts) > model_index + 1:
                version = int(parts[model_index + 1])
        except ValueError:
            pass

    return workspace, project, version


def download_roboflow_dataset(url, output_dir, api_key=None):
    """
    Download dataset from Roboflow
    URL format: https://universe.roboflow.com/workspace/project/...

    Use roboflow CLI or API
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        from roboflow import Roboflow

        if not api_key:
            raise ValueError("API key required. Get it from: https://app.roboflow.com/settings/api")

        rf = Roboflow(api_key=api_key)

        workspace, project, version_num = extract_roboflow_details(url)

        if not workspace or not project:
            print(f"Error: Could not parse URL: {url}")
            return None

        print(f"Workspace: {workspace}, Project: {project}, Version: {version_num}")

        project_obj = rf.workspace(workspace).project(project)

        if version_num:
            print(f"Downloading specific version: {version_num}")
            version = project_obj.version(version_num)
        else:
            print("Fetching latest version...")
            versions = project_obj.versions()
            if not versions:
                print(f"No versions found for {project}")
                return None
            version = versions[-1]
            print(f"Auto-selected version: {version.version}")

        dataset = version.download("coco", location=str(output_dir))

        print(f"Downloaded to: {output_dir}")
        return output_dir

    except ImportError:
        print("Roboflow library not installed. Install with: pip install roboflow")
        print(f"Manual download: Visit {url} and download COCO format")
        return None


def filter_images_with_multiple_objects(coco_json_path, min_objects=2):
    """
    Filter images that have >= min_objects instances
    Returns: dict of image_id -> num_objects
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    image_object_count = defaultdict(int)

    for ann in coco_data['annotations']:
        image_object_count[ann['image_id']] += 1

    filtered_images = {
        img_id: count
        for img_id, count in image_object_count.items()
        if count >= min_objects
    }

    print(f"Total images: {len(coco_data['images'])}")
    print(f"Images with >={min_objects} objects: {len(filtered_images)}")

    return filtered_images, coco_data


def select_exemplars(annotations, num_exemplars=3):
    """
    Select exemplar bounding boxes from annotations
    Strategy: select boxes from different spatial regions
    """
    if len(annotations) < num_exemplars:
        return [ann['bbox'] for ann in annotations]

    sorted_anns = sorted(annotations, key=lambda x: x['bbox'][0])

    step = len(sorted_anns) // num_exemplars
    exemplars = []

    for i in range(num_exemplars):
        idx = i * step
        bbox = sorted_anns[idx]['bbox']
        x, y, w, h = bbox
        exemplars.append([int(x), int(y), int(x + w), int(y + h)])

    return exemplars


def coco_to_countgd_odvg(
    coco_json_path,
    images_dir,
    output_jsonl_path,
    category_name,
    min_objects=2,
    num_exemplars=3
):
    """
    Convert COCO format to CountGD ODVG format (JSONL)

    CountGD ODVG format:
    {
        "filename": "image.jpg",
        "height": 384,
        "width": 512,
        "detection": {
            "instances": [
                {"bbox": [x1, y1, x2, y2], "label": 0, "category": "object_name"},
                ...
            ]
        },
        "exemplars": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    }
    """
    filtered_images, coco_data = filter_images_with_multiple_objects(coco_json_path, min_objects)

    image_id_to_filename = {img['id']: img for img in coco_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['image_id'] in filtered_images:
            image_annotations[ann['image_id']].append(ann)

    output_data = []

    for img_id, anns in tqdm(image_annotations.items(), desc="Converting"):
        img_info = image_id_to_filename[img_id]

        instances = []
        for ann in anns:
            x, y, w, h = ann['bbox']

            cx = x + w / 2
            cy = y + h / 2

            instances.append({
                "bbox": [cx - 1, cy - 1, cx + 1, cy + 1],
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

        output_data.append(entry)

    with open(output_jsonl_path, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(output_data)} entries to {output_jsonl_path}")

    return output_data


def create_label_map(category_names, output_path):
    """
    Create label map JSON
    Format: {"0": "category_name", "1": "category_name_2", ...}
    """
    label_map = {str(i): name for i, name in enumerate(category_names)}

    with open(output_path, 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"Created label map: {output_path}")
    return label_map


def analyze_dataset(jsonl_path, images_dir):
    """
    Analyze converted dataset and generate statistics
    """
    entries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line))

    stats = {
        'total_images': len(entries),
        'total_objects': 0,
        'objects_per_image': [],
        'image_sizes': [],
        'categories': defaultdict(int),
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
    print(f"Average objects per image: {np.mean(stats['objects_per_image']):.2f}")
    print(f"Min objects per image: {np.min(stats['objects_per_image'])}")
    print(f"Max objects per image: {np.max(stats['objects_per_image'])}")
    print(f"Std objects per image: {np.std(stats['objects_per_image']):.2f}")

    print(f"\nCategory distribution:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count} instances")

    print(f"\nImage size statistics:")
    widths = [s[0] for s in stats['image_sizes']]
    heights = [s[1] for s in stats['image_sizes']]
    print(f"  Width - Min: {np.min(widths)}, Max: {np.max(widths)}, Avg: {np.mean(widths):.1f}")
    print(f"  Height - Min: {np.min(heights)}, Max: {np.max(heights)}, Avg: {np.mean(heights):.1f}")
    print("="*80)

    return stats


def process_multiple_roboflow_datasets(dataset_configs, output_dir, api_key=None):
    """
    Process multiple Roboflow datasets

    dataset_configs format:
    [
        {"url": "https://universe.roboflow.com/...", "category": "pipe"},
        {"url": "https://universe.roboflow.com/...", "category": "screw"},
    ]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_train_data = []
    all_val_data = []
    all_categories = []

    for config in dataset_configs:
        url = config['url']
        category = config['category']
        min_objects = config.get('min_objects', 2)

        print(f"\n{'='*80}")
        print(f"Processing: {category}")
        print(f"URL: {url}")
        print(f"{'='*80}")

        dataset_dir = output_dir / f"raw_{category}"

        downloaded = download_roboflow_dataset(url, dataset_dir, api_key)

        if not downloaded:
            print(f"Skipping {category} - download failed")
            continue

        train_json = dataset_dir / "train" / "_annotations.coco.json"
        val_json = dataset_dir / "valid" / "_annotations.coco.json"

        train_images = dataset_dir / "train"
        val_images = dataset_dir / "valid"

        if train_json.exists():
            train_jsonl = output_dir / f"{category}_train.jsonl"
            train_data = coco_to_countgd_odvg(
                train_json,
                train_images,
                train_jsonl,
                category,
                min_objects=min_objects
            )
            all_train_data.extend(train_data)

        if val_json.exists():
            val_jsonl = output_dir / f"{category}_val.jsonl"
            val_data = coco_to_countgd_odvg(
                val_json,
                val_images,
                val_jsonl,
                category,
                min_objects=min_objects
            )
            all_val_data.extend(val_data)

        all_categories.append(category)

        shutil.copytree(train_images, output_dir / "images" / category / "train", dirs_exist_ok=True)
        if val_images.exists():
            shutil.copytree(val_images, output_dir / "images" / category / "valid", dirs_exist_ok=True)

    combined_train_jsonl = output_dir / "combined_train.jsonl"
    with open(combined_train_jsonl, 'w') as f:
        for entry in all_train_data:
            f.write(json.dumps(entry) + '\n')

    combined_val_jsonl = output_dir / "combined_val.jsonl"
    with open(combined_val_jsonl, 'w') as f:
        for entry in all_val_data:
            f.write(json.dumps(entry) + '\n')

    label_map_path = output_dir / "label_map.json"
    create_label_map(all_categories, label_map_path)

    print("\n" + "="*80)
    print("COMBINED DATASET ANALYSIS - TRAIN")
    print("="*80)
    analyze_dataset(combined_train_jsonl, output_dir / "images")

    if all_val_data:
        print("\n" + "="*80)
        print("COMBINED DATASET ANALYSIS - VAL")
        print("="*80)
        analyze_dataset(combined_val_jsonl, output_dir / "images")

    dataset_config = {
        "train": [
            {
                "root": str((output_dir / "images").absolute()),
                "anno": str(combined_train_jsonl.absolute()),
                "label_map": str(label_map_path.absolute()),
                "dataset_mode": "odvg"
            }
        ],
        "val": [
            {
                "root": str((output_dir / "images").absolute()),
                "anno": str(combined_val_jsonl.absolute()),
                "label_map": None,
                "dataset_mode": "odvg"
            }
        ]
    }

    dataset_config_path = output_dir / "dataset_config.json"
    with open(dataset_config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Dataset config saved to: {dataset_config_path}")
    print(f"Use this in train.sh with --datasets {dataset_config_path}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser("Convert Roboflow datasets to CountGD format")
    parser.add_argument('--config', type=str, required=True, help='JSON file with dataset configs')
    parser.add_argument('--output_dir', type=str, default='countgd_data', help='Output directory')
    parser.add_argument('--api_key', type=str, help='Roboflow API key')
    parser.add_argument('--min_objects', type=int, default=2, help='Minimum objects per image')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dataset_configs = json.load(f)

    for config in dataset_configs:
        config.setdefault('min_objects', args.min_objects)

    process_multiple_roboflow_datasets(
        dataset_configs,
        args.output_dir,
        api_key=args.api_key
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
