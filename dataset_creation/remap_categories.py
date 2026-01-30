import json
import argparse
from pathlib import Path
from collections import defaultdict


def remap_categories_in_jsonl(jsonl_path, category_mapping, output_path=None):
    """
    Remap category names in a JSONL file

    Args:
        jsonl_path: Path to input JSONL
        category_mapping: Dict mapping old names to new names
        output_path: Output path (if None, overwrites input)
    """
    entries = []
    remapping_stats = defaultdict(int)

    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)

            for inst in entry['detection']['instances']:
                old_category = inst['category']

                if old_category in category_mapping:
                    new_category = category_mapping[old_category]
                    inst['category'] = new_category
                    remapping_stats[f"{old_category} -> {new_category}"] += 1
                else:
                    remapping_stats[f"{old_category} (unchanged)"] += 1

            entries.append(entry)

    if output_path is None:
        output_path = jsonl_path

    with open(output_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    return remapping_stats


def update_label_map(label_map_path, category_mapping):
    """Update label_map.json with new category names"""
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)

    new_label_map = {}

    for idx, old_name in label_map.items():
        if old_name in category_mapping:
            new_label_map[idx] = category_mapping[old_name]
        else:
            new_label_map[idx] = old_name

    unique_categories = sorted(set(new_label_map.values()))
    final_label_map = {str(i): cat for i, cat in enumerate(unique_categories)}

    with open(label_map_path, 'w') as f:
        json.dump(final_label_map, f, indent=2)

    print(f"\n📝 Updated label map:")
    for idx, cat in final_label_map.items():
        print(f"  {idx}: {cat}")

    return final_label_map


def analyze_categories(jsonl_paths):
    """Analyze current categories in JSONL files"""
    category_counts = defaultdict(int)

    for jsonl_path in jsonl_paths:
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                for inst in entry['detection']['instances']:
                    category_counts[inst['category']] += 1

    print("\n📊 Current category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:40s}: {count:6d} instances")

    return category_counts


def main():
    parser = argparse.ArgumentParser("Remap category names in CountGD dataset")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to final data directory')
    parser.add_argument('--mapping', type=str, required=True, help='JSON file with category mapping')
    parser.add_argument('--dry_run', action='store_true', help='Show what would change without applying')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    with open(args.mapping, 'r') as f:
        category_mapping = json.load(f)

    print("="*80)
    print("CATEGORY REMAPPING")
    print("="*80)
    print("\nMapping:")
    for old, new in category_mapping.items():
        print(f"  {old:40s} -> {new}")

    jsonl_files = [
        data_dir / "train.jsonl",
        data_dir / "val.jsonl",
        data_dir / "test.jsonl"
    ]

    existing_jsonl = [f for f in jsonl_files if f.exists()]

    if not existing_jsonl:
        print(f"\n❌ No JSONL files found in {data_dir}")
        return

    print(f"\nBefore remapping:")
    before_stats = analyze_categories(existing_jsonl)

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")
        print("\nWould remap:")
        for old, new in category_mapping.items():
            if old in before_stats:
                print(f"  {old:40s} -> {new:40s} ({before_stats[old]} instances)")
        return

    print("\n" + "="*80)
    print("APPLYING REMAPPING")
    print("="*80)

    all_stats = defaultdict(int)

    for jsonl_path in existing_jsonl:
        print(f"\nProcessing: {jsonl_path.name}")
        stats = remap_categories_in_jsonl(jsonl_path, category_mapping)

        for key, count in stats.items():
            all_stats[key] += count

    print("\n📊 Remapping summary:")
    for mapping, count in sorted(all_stats.items()):
        print(f"  {mapping:60s}: {count:6d} instances")

    label_map_path = data_dir / "label_map.json"
    if label_map_path.exists():
        update_label_map(label_map_path, category_mapping)

    print(f"\nAfter remapping:")
    after_stats = analyze_categories(existing_jsonl)

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Before: {len(before_stats)} unique categories")
    print(f"After:  {len(after_stats)} unique categories")

    print("\n✅ Remapping complete!")
    print(f"\nUpdated files:")
    for f in existing_jsonl:
        print(f"  - {f}")
    print(f"  - {label_map_path}")


if __name__ == '__main__':
    main()
