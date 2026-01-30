from roboflow import Roboflow
import json
from pathlib import Path
import shutil
import argparse
from manual_coco_converter import coco_to_countgd_manual, analyze_converted_data
from tqdm import tqdm
from collections import defaultdict


def extract_roboflow_details(url):
    """Parse Roboflow URL"""
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


def download_single_dataset(rf, url, output_base_dir):
    """Download a single dataset from Roboflow"""
    workspace, project_id, version_num = extract_roboflow_details(url)

    if not workspace or not project_id:
        print(f"⚠️  Invalid URL: {url}")
        return None

    try:
        print(f"🔍 Accessing: {workspace}/{project_id}...")
        project = rf.workspace(workspace).project(project_id)

        if version_num:
            print(f"   Version: {version_num}")
            version = project.version(version_num)
        else:
            versions = project.versions()
            if not versions:
                print(f"   ❌ No versions found")
                return None
            version = versions[-1]
            print(f"   Auto-selected version: {version.version}")

        download_dir = output_base_dir / f"{workspace}_{project_id}"
        print(f"   ⬇️  Downloading to {download_dir}...")
        dataset = version.download("coco", location=str(download_dir))

        return {
            'workspace': workspace,
            'project': project_id,
            'version': version.version,
            'download_dir': download_dir,
            'location': dataset.location
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def convert_downloaded_dataset(dataset_info, category_name, min_objects=2, merge_train_val=True):
    """Convert downloaded COCO dataset to CountGD format"""
    download_dir = Path(dataset_info['download_dir'])

    train_json = download_dir / "train" / "_annotations.coco.json"
    val_json = download_dir / "valid" / "_annotations.coco.json"

    train_images = download_dir / "train"
    val_images = download_dir / "valid"

    all_entries = []

    if train_json.exists():
        print(f"   Converting train set...")
        train_jsonl = download_dir / "train_countgd.jsonl"
        train_entries, train_stats = coco_to_countgd_manual(
            train_json,
            train_jsonl,
            category_filter=None,
            min_objects=min_objects,
            num_exemplars=3,
            point_annotation=True
        )

        for entry in train_entries:
            entry['_source_split'] = 'train'
            entry['_source_dir'] = train_images

        all_entries.extend(train_entries)

    if val_json.exists():
        print(f"   Converting val set...")
        val_jsonl = download_dir / "val_countgd.jsonl"
        val_entries, val_stats = coco_to_countgd_manual(
            val_json,
            val_jsonl,
            category_filter=None,
            min_objects=min_objects,
            num_exemplars=3,
            point_annotation=True
        )

        for entry in val_entries:
            entry['_source_split'] = 'valid'
            entry['_source_dir'] = val_images

        all_entries.extend(val_entries)

    print(f"   Total entries from this dataset: {len(all_entries)}")

    return {'all': all_entries}


def merge_datasets(all_converted_data, output_dir, category_mapping, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Merge multiple datasets and create train/val/test splits from all data"""
    import random
    random.seed(random_seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_entries = []
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(exist_ok=True, parents=True)

    for idx, data in enumerate(all_converted_data):
        dataset_name = data['dataset_info']['project']
        category_name = data.get('category_name', dataset_name)

        if 'converted' in data and 'all' in data['converted']:
            entries = data['converted']['all']

            dataset_img_dir = images_output_dir / f"dataset_{idx}"
            dataset_img_dir.mkdir(exist_ok=True, parents=True)

            for entry in entries:
                for inst in entry['detection']['instances']:
                    inst['category'] = category_name

                source_dir = entry.pop('_source_dir')
                entry.pop('_source_split', None)

                src_img = Path(source_dir) / entry['filename']
                if src_img.exists():
                    dst_img = dataset_img_dir / entry['filename']
                    dst_img.parent.mkdir(exist_ok=True, parents=True)
                    shutil.copy2(src_img, dst_img)

                    entry['filename'] = str(Path(f"dataset_{idx}") / entry['filename'])

            all_entries.extend(entries)

    print(f"\n📊 Total merged entries: {len(all_entries)}")

    random.shuffle(all_entries)

    total = len(all_entries)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    all_train_entries = all_entries[:train_size]
    all_val_entries = all_entries[train_size:train_size + val_size]
    all_test_entries = all_entries[train_size + val_size:]

    print(f"Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    print(f"Train: {len(all_train_entries)} images")
    print(f"Val: {len(all_val_entries)} images")
    print(f"Test: {len(all_test_entries)} images")

    train_jsonl = output_dir / "train.jsonl"
    with open(train_jsonl, 'w') as f:
        for entry in all_train_entries:
            f.write(json.dumps(entry) + '\n')

    val_jsonl = output_dir / "val.jsonl"
    with open(val_jsonl, 'w') as f:
        for entry in all_val_entries:
            f.write(json.dumps(entry) + '\n')

    test_jsonl = None
    if all_test_entries:
        test_jsonl = output_dir / "test.jsonl"
        with open(test_jsonl, 'w') as f:
            for entry in all_test_entries:
                f.write(json.dumps(entry) + '\n')

    all_categories = set()
    for entry in all_train_entries + all_val_entries + all_test_entries:
        for inst in entry['detection']['instances']:
            all_categories.add(inst['category'])

    label_map = {str(i): cat for i, cat in enumerate(sorted(all_categories))}
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)

    dataset_config = {
        "train": [
            {
                "root": str(images_output_dir.absolute()),
                "anno": str(train_jsonl.absolute()),
                "label_map": str(label_map_path.absolute()),
                "dataset_mode": "odvg"
            }
        ],
        "val": [
            {
                "root": str(images_output_dir.absolute()),
                "anno": str(val_jsonl.absolute()),
                "label_map": None,
                "dataset_mode": "odvg"
            }
        ]
    }

    if test_jsonl:
        dataset_config["test"] = [
            {
                "root": str(images_output_dir.absolute()),
                "anno": str(test_jsonl.absolute()),
                "label_map": None,
                "dataset_mode": "odvg"
            }
        ]

    config_path = output_dir / "dataset_config.json"
    with open(config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)

    print(f"\n{'='*80}")
    print(f"MERGED DATASET")
    print(f"{'='*80}")
    print(f"Train: {len(all_train_entries)} images")
    print(f"Val: {len(all_val_entries)} images")
    if all_test_entries:
        print(f"Test: {len(all_test_entries)} images")
    print(f"Categories: {', '.join(sorted(all_categories))}")
    print(f"\nFiles:")
    print(f"  Config: {config_path}")
    print(f"  Train: {train_jsonl}")
    print(f"  Val: {val_jsonl}")
    if test_jsonl:
        print(f"  Test: {test_jsonl}")
    print(f"  Images: {images_output_dir}")
    print(f"{'='*80}")

    return config_path


def main():
    parser = argparse.ArgumentParser("Download Roboflow datasets and convert to CountGD")
    parser.add_argument('--config', type=str, required=True, help='JSON config with URLs')
    parser.add_argument('--api_key', type=str, required=True, help='Roboflow API key')
    parser.add_argument('--output_dir', type=str, default='roboflow_data')
    parser.add_argument('--min_objects', type=int, default=2)
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Val split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for splitting')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    rf = Roboflow(api_key=args.api_key)

    output_base = Path(args.output_dir) / "downloads"
    output_base.mkdir(exist_ok=True, parents=True)

    all_converted_data = []

    print(f"Starting download for {len(config)} datasets...\n")

    for item in config:
        url = item['url']
        category_name = item.get('category', 'object')
        min_objects = item.get('min_objects', args.min_objects)

        print(f"\n{'='*80}")
        print(f"Processing: {category_name}")
        print(f"URL: {url}")
        print(f"{'='*80}")

        dataset_info = download_single_dataset(rf, url, output_base)

        if not dataset_info:
            continue

        converted = convert_downloaded_dataset(dataset_info, category_name, min_objects)

        all_converted_data.append({
            'dataset_info': dataset_info,
            'category_name': category_name,
            'converted': converted
        })

    if all_converted_data:
        category_mapping = {item['dataset_info']['project']: item['category_name'] for item in all_converted_data}
        final_output = Path(args.output_dir) / "final"
        config_path = merge_datasets(
            all_converted_data,
            final_output,
            category_mapping,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )

        print(f"\n{'='*80}")
        print("CREATING COCO FORMAT FOR VAL/TEST")
        print(f"{'='*80}")

        from custom_dataset.convert_to_coco import countgd_odvg_to_coco, update_dataset_config

        images_dir = final_output / "images"

        val_jsonl = final_output / "val.jsonl"
        if val_jsonl.exists():
            print("\nConverting val to COCO...")
            val_coco_path = final_output / "val_coco.json"
            countgd_odvg_to_coco(val_jsonl, images_dir, val_coco_path, category_mapping)

        test_jsonl = final_output / "test.jsonl"
        test_coco_path = None
        if test_jsonl.exists():
            print("\nConverting test to COCO...")
            test_coco_path = final_output / "test_coco.json"
            countgd_odvg_to_coco(test_jsonl, images_dir, test_coco_path, category_mapping)

        print("\nUpdating dataset config...")
        final_config = update_dataset_config(
            config_path,
            val_coco_path.absolute() if val_jsonl.exists() else None,
            test_coco_path.absolute() if test_coco_path else None,
            images_dir.absolute()
        )

        print(f"\n✅ All done!")
        print(f"\nUse this config for training:")
        print(f"  --datasets {final_config}")


if __name__ == '__main__':
    main()
