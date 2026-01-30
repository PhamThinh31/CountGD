"""
Map raw YOLO category names to descriptive construction text categories.

CountGD uses text prompts to count objects, so the category name must be
a natural language description of the object, not an internal part code.

Multiple raw codes that refer to visually identical parts share the same
text category so the model learns a single concept per visual appearance.

Usage:
    # As a library
    from category_text_mapping import get_text_category, remap_odvg_categories

    text = get_text_category("A-203")          # "scaffolding frame"
    remap_odvg_categories("train.jsonl", "train_remapped.jsonl")

    # As a CLI
    python category_text_mapping.py --input train.jsonl --output train_remapped.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

# ─── Raw category -> text description ─────────────────────────────────────────
#
# Grouping logic based on visual inspection of actual images:
#
#   scaffolding frame         Flat H-shaped or rectangular steel frames
#                             with connection holes along the edges.
#                             (A-series alloy designations: A-2, A-203, etc.)
#
#   scaffolding bracket       Hook-shaped metal connectors that attach
#                             horizontal ledgers to vertical standards.
#                             (BKN-series: BKN-2 through BKN-624)
#
#   scaffolding coupler       Cylindrical clamp devices that join two
#                             scaffold tubes together at fixed angles.
#                             (IQC-series: IQC305, IQC0914, IQC1219, IQC1524)
#
#   scaffolding nut           Threaded fastener used to lock couplers,
#                             base jacks, and other adjustable parts.
#                             (IQN-series: IQN2 through IQN624)
#
#   scaffolding base jack     Adjustable vertical screw assembly seated
#                             at the bottom of scaffold standards to
#                             level the structure on uneven ground.
#                             (IQA-series: IQA238A, IQA950, IQA3800)
#
#   scaffolding steel pipe    Hollow round steel tubes used as standards,
#                             ledgers, or transoms in a scaffold system.
#                             (PIPE-series: P20, P25, P30, P45)
#
#   scaffolding jack base     Threaded base plate with a vertical stem,
#                             used as the lowest support point.
#                             (S-series: S2, S4)
#
#   scaffolding swivel        Rotatable coupler that allows angular
#                             connection between two tubes.
#                             (SW-series: SW2, SW4)
#
#   scaffolding ledger        Horizontal tube with end-fittings that
#                             clicks into rosettes on vertical standards.
#                             (L-series: L2, L4)
#
#   scaffolding crossbar      Diagonal or cross-shaped bracing member
#                             that stiffens the scaffold frame.
#                             (Type6)
# ──────────────────────────────────────────────────────────────────────────────

CATEGORY_TEXT_MAP = {
    # ── A-series: scaffolding frames ──
    "A-2":      "scaffolding frame",
    "A-203":    "scaffolding frame",
    "A-217":    "scaffolding frame",
    "A-303L":   "scaffolding frame",
    "A-304L":   "scaffolding frame",
    "A-3055A":  "scaffolding frame",
    "A-403L":   "scaffolding frame",

    # ── BKN-series: scaffolding brackets ──
    "BKN-2":    "scaffolding bracket",
    "BKN-3":    "scaffolding bracket",
    "BKN-4":    "scaffolding bracket",
    "BKN-5":    "scaffolding bracket",
    "BKN-224":  "scaffolding bracket",
    "BKN-324":  "scaffolding bracket",
    "BKN-524":  "scaffolding bracket",
    "BKN-624":  "scaffolding bracket",
    "BKN-6S":   "scaffolding bracket",

    # ── IQC-series: scaffolding couplers ──
    "IQC305":   "scaffolding coupler",
    "IQC0914":  "scaffolding coupler",
    "IQC1219":  "scaffolding coupler",
    "IQC1524":  "scaffolding coupler",

    # ── IQN-series: scaffolding nuts ──
    "IQN2":     "scaffolding nut",
    "IQN3":     "scaffolding nut",
    "IQN4":     "scaffolding nut",
    "IQN5":     "scaffolding nut",
    "IQN324":   "scaffolding nut",
    "IQN424":   "scaffolding nut",
    "IQN524":   "scaffolding nut",
    "IQN624":   "scaffolding nut",

    # ── IQA-series: scaffolding base jacks / assemblies ──
    "IQA238A":  "scaffolding base jack",
    "IQA950":   "scaffolding base jack",
    "IQA3800":  "scaffolding base jack",

    # ── PIPE-series: scaffolding steel pipes ──
    "P20":      "scaffolding steel pipe",
    "P25":      "scaffolding steel pipe",
    "P30":      "scaffolding steel pipe",
    "P45":      "scaffolding steel pipe",

    # ── S-series: scaffolding jack bases ──
    "S2":       "scaffolding jack base",
    "S4":       "scaffolding jack base",

    # ── SW-series: scaffolding swivels ──
    "SW2":      "scaffolding swivel",
    "SW4":      "scaffolding swivel",

    # ── L-series: scaffolding ledgers ──
    "L2":       "scaffolding ledger",
    "L4":       "scaffolding ledger",

    # ── Type6: scaffolding crossbar ──
    "Type6":    "scaffolding crossbar",
}

# Reverse lookup: text category -> list of raw codes
TEXT_TO_RAW = defaultdict(list)
for _raw, _text in CATEGORY_TEXT_MAP.items():
    TEXT_TO_RAW[_text].append(_raw)


def get_text_category(raw_name):
    """Return the text category for a raw category name.
    Falls back to lowercase raw_name if not mapped.
    """
    return CATEGORY_TEXT_MAP.get(raw_name, raw_name.lower())


def remap_odvg_categories(input_jsonl, output_jsonl):
    """Rewrite an ODVG JSONL file replacing raw category names with text categories.

    Also rewrites label IDs so that entries sharing the same text category
    get the same label number.

    Args:
        input_jsonl:  Path to source .jsonl
        output_jsonl: Path to write remapped .jsonl

    Returns:
        dict with:
            text_label_map: {str(id): text_category}
            stats: {text_category: {"images": int, "annotations": int, "raw_codes": set}}
    """
    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)

    # Build text -> label_id
    unique_texts = sorted(set(
        get_text_category(raw) for raw in CATEGORY_TEXT_MAP.values()
    ))
    text_to_label = {t: i for i, t in enumerate(unique_texts)}

    stats = defaultdict(lambda: {"images": 0, "annotations": 0, "raw_codes": set()})

    with open(input_jsonl, "r") as fin, open(output_jsonl, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            seen_texts = set()

            for inst in entry["detection"]["instances"]:
                raw = inst["category"]
                text = get_text_category(raw)
                inst["category"] = text
                inst["label"] = text_to_label.get(text, 0)
                stats[text]["annotations"] += 1
                stats[text]["raw_codes"].add(raw)
                seen_texts.add(text)

            for text in seen_texts:
                stats[text]["images"] += 1

            fout.write(json.dumps(entry) + "\n")

    # label map
    text_label_map = {str(v): k for k, v in text_to_label.items()}

    return {"text_label_map": text_label_map, "stats": stats}


def remap_coco_categories(input_json, output_json):
    """Rewrite a COCO JSON file replacing raw category names with text categories.

    Args:
        input_json:  Path to source COCO .json
        output_json: Path to write remapped .json

    Returns:
        dict with text_label_map and stats
    """
    input_json = Path(input_json)
    output_json = Path(output_json)

    with open(input_json, "r") as f:
        data = json.load(f)

    # Build old_id -> raw_name
    old_id_to_raw = {}
    for cat in data.get("categories", []):
        old_id_to_raw[cat["id"]] = cat["name"]

    # Build text categories and new IDs
    unique_texts = sorted(set(
        get_text_category(raw) for raw in old_id_to_raw.values()
    ))
    text_to_new_id = {t: i for i, t in enumerate(unique_texts)}

    # Old category id -> new category id
    old_to_new = {}
    for old_id, raw_name in old_id_to_raw.items():
        text = get_text_category(raw_name)
        old_to_new[old_id] = text_to_new_id[text]

    # Rewrite categories
    data["categories"] = [{"id": i, "name": t} for t, i in text_to_new_id.items()]

    # Rewrite annotations
    for ann in data.get("annotations", []):
        ann["category_id"] = old_to_new.get(ann["category_id"], ann["category_id"])

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    text_label_map = {str(v): k for k, v in text_to_new_id.items()}
    return {"text_label_map": text_label_map}


def remap_full_output(output_dir):
    """Remap all files in a yolo_to_countgd output directory in place.

    Expects:
        output_dir/
            train.jsonl
            val.jsonl        (or val_coco.json)
            test.jsonl       (or test_coco.json)
            label_map.json
            dataset_config.json
            dataset_config_mixed.json

    Creates remapped versions alongside originals:
        output_dir/
            train.jsonl              <- overwritten with text categories
            val.jsonl                <- overwritten
            test.jsonl               <- overwritten
            val_coco.json            <- overwritten
            test_coco.json           <- overwritten
            label_map.json           <- overwritten with text label map
            category_text_mapping.json <- full raw->text mapping saved
    """
    output_dir = Path(output_dir)

    print("=" * 70)
    print("Remapping categories to construction text descriptions")
    print("=" * 70)

    result_stats = {}

    # ODVG files
    for name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        p = output_dir / name
        if not p.exists():
            continue
        tmp = p.with_suffix(".jsonl.tmp")
        print(f"\n  Remapping {name}...")
        result = remap_odvg_categories(p, tmp)
        tmp.rename(p)

        for text, s in result["stats"].items():
            if text not in result_stats:
                result_stats[text] = {"images": 0, "annotations": 0, "raw_codes": set()}
            result_stats[text]["images"] += s["images"]
            result_stats[text]["annotations"] += s["annotations"]
            result_stats[text]["raw_codes"] |= s["raw_codes"]

        text_label_map = result["text_label_map"]

    # COCO files
    for name in ["val_coco.json", "test_coco.json"]:
        p = output_dir / name
        if not p.exists():
            continue
        tmp = p.with_suffix(".json.tmp")
        print(f"  Remapping {name}...")
        remap_coco_categories(p, tmp)
        tmp.rename(p)

    # Overwrite label_map
    label_map_path = output_dir / "label_map.json"
    if text_label_map:
        with open(label_map_path, "w") as f:
            json.dump(text_label_map, f, indent=2)
        print(f"  Updated {label_map_path}")

    # Save full mapping for reference
    mapping_path = output_dir / "category_text_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(CATEGORY_TEXT_MAP, f, indent=2)
    print(f"  Saved mapping -> {mapping_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"{'Text Category':<30} {'Images':>8} {'Annots':>10}   Raw Codes")
    print(f"{'-' * 70}")
    for text in sorted(result_stats):
        s = result_stats[text]
        codes = ", ".join(sorted(s["raw_codes"]))
        print(f"  {text:<28} {s['images']:>8,} {s['annotations']:>10,}   {codes}")
    print(f"{'=' * 70}")
    print(f"  Total text categories: {len(result_stats)}")
    print(f"  Total raw codes mapped: {sum(len(s['raw_codes']) for s in result_stats.values())}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Remap raw category codes to construction text descriptions"
    )
    sub = parser.add_subparsers(dest="command")

    # Single file
    sp = sub.add_parser("file", help="Remap a single JSONL or COCO JSON file")
    sp.add_argument("--input", required=True, help="Input file path")
    sp.add_argument("--output", required=True, help="Output file path")
    sp.add_argument("--format", choices=["odvg", "coco"], default="odvg")

    # Full output dir
    sp2 = sub.add_parser("dir", help="Remap all files in a yolo_to_countgd output dir")
    sp2.add_argument("--path", required=True, help="Path to output directory")

    # Print mapping
    sub.add_parser("show", help="Print the category text mapping table")

    args = parser.parse_args()

    if args.command == "file":
        if args.format == "odvg":
            result = remap_odvg_categories(args.input, args.output)
        else:
            result = remap_coco_categories(args.input, args.output)
        print(json.dumps(result["text_label_map"], indent=2))

    elif args.command == "dir":
        remap_full_output(args.path)

    elif args.command == "show":
        print(f"\n{'Raw Code':<15} {'Text Category':<30}")
        print("-" * 45)
        for raw in sorted(CATEGORY_TEXT_MAP):
            print(f"  {raw:<13} {CATEGORY_TEXT_MAP[raw]}")
        print(f"\nUnique text categories: {len(set(CATEGORY_TEXT_MAP.values()))}")
        print(f"Total raw codes: {len(CATEGORY_TEXT_MAP)}")
        print(f"\nText categories:")
        for text in sorted(set(CATEGORY_TEXT_MAP.values())):
            codes = [r for r, t in CATEGORY_TEXT_MAP.items() if t == text]
            print(f"  {text}: {', '.join(sorted(codes))}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
