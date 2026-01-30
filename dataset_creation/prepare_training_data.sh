#!/bin/bash

set -e

MODE=$1
CONFIG_FILE=${2:-"dataset_configs_example.json"}
OUTPUT_DIR=${3:-"countgd_training_data"}
API_KEY=${4}

case $MODE in

  "download")
    echo "=== DOWNLOADING ROBOFLOW DATASETS ==="

    if [ -z "$API_KEY" ]; then
      echo "Error: API key required for download"
      echo "Usage: $0 download <config.json> <output_dir> <roboflow_api_key>"
      echo ""
      echo "Get your API key from: https://app.roboflow.com/settings/api"
      exit 1
    fi

    pip install roboflow

    python roboflow_to_countgd.py \
      --config "$CONFIG_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --api_key "$API_KEY" \
      --min_objects 2

    echo "Download complete!"
    ;;

  "convert")
    echo "=== CONVERTING MANUAL COCO DATASET ==="

    COCO_JSON=${2}
    OUTPUT_JSONL=${3:-"converted_data.jsonl"}

    if [ -z "$COCO_JSON" ]; then
      echo "Usage: $0 convert <coco_json> <output_jsonl> [category_filter...]"
      exit 1
    fi

    shift 3 2>/dev/null || shift 2 2>/dev/null || true
    CATEGORIES="$@"

    if [ -n "$CATEGORIES" ]; then
      python manual_coco_converter.py \
        --coco_json "$COCO_JSON" \
        --output_jsonl "$OUTPUT_JSONL" \
        --category_filter $CATEGORIES \
        --min_objects 2 \
        --point_annotation \
        --split \
        --split_ratio 0.8
    else
      python manual_coco_converter.py \
        --coco_json "$COCO_JSON" \
        --output_jsonl "$OUTPUT_JSONL" \
        --min_objects 2 \
        --point_annotation \
        --split \
        --split_ratio 0.8
    fi

    echo "Conversion complete!"
    ;;

  "analyze")
    echo "=== ANALYZING DATASET ==="

    JSONL_FILE=${2}

    if [ -z "$JSONL_FILE" ]; then
      echo "Usage: $0 analyze <jsonl_file>"
      exit 1
    fi

    python -c "
import json
import numpy as np
from collections import defaultdict

entries = []
with open('$JSONL_FILE', 'r') as f:
    for line in f:
        entries.append(json.loads(line))

print('='*80)
print('DATASET ANALYSIS')
print('='*80)
print(f'Total images: {len(entries)}')

objects_per_image = [len(e['detection']['instances']) for e in entries]
print(f'Total objects: {sum(objects_per_image)}')
print(f'Avg objects/image: {np.mean(objects_per_image):.2f} ± {np.std(objects_per_image):.2f}')
print(f'Min/Max: {np.min(objects_per_image)} / {np.max(objects_per_image)}')

categories = defaultdict(int)
for e in entries:
    for inst in e['detection']['instances']:
        categories[inst['category']] += 1

print(f'\nCategories:')
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f'  {cat}: {count}')

widths = [e['width'] for e in entries]
heights = [e['height'] for e in entries]
print(f'\nImage sizes:')
print(f'  Width: {np.min(widths)} - {np.max(widths)} (avg: {np.mean(widths):.1f})')
print(f'  Height: {np.min(heights)} - {np.max(heights)} (avg: {np.mean(heights):.1f})')

print('='*80)
"
    ;;

  "visualize")
    echo "=== VISUALIZING SAMPLES ==="

    JSONL_FILE=${2}
    IMAGES_DIR=${3}
    NUM_SAMPLES=${4:-5}

    if [ -z "$JSONL_FILE" ] || [ -z "$IMAGES_DIR" ]; then
      echo "Usage: $0 visualize <jsonl_file> <images_dir> [num_samples]"
      exit 1
    fi

    python visualize_countgd_data.py \
      --jsonl "$JSONL_FILE" \
      --images_dir "$IMAGES_DIR" \
      --num_samples "$NUM_SAMPLES" \
      --output_dir visualizations

    echo "Visualizations saved to visualizations/"
    ;;

  *)
    echo "CountGD Training Data Preparation Pipeline"
    echo ""
    echo "Usage: $0 {download|convert|analyze|visualize} [args...]"
    echo ""
    echo "Commands:"
    echo "  download  - Download from Roboflow and convert to CountGD format"
    echo "              $0 download <config.json> <output_dir> <api_key>"
    echo ""
    echo "  convert   - Convert existing COCO dataset to CountGD format"
    echo "              $0 convert <coco.json> <output.jsonl> [category1 category2 ...]"
    echo ""
    echo "  analyze   - Analyze converted JSONL dataset"
    echo "              $0 analyze <data.jsonl>"
    echo ""
    echo "  visualize - Visualize dataset samples"
    echo "              $0 visualize <data.jsonl> <images_dir> [num_samples]"
    echo ""
    echo "Examples:"
    echo "  # Download and convert Roboflow datasets"
    echo "  $0 download dataset_configs.json my_data YOUR_API_KEY"
    echo ""
    echo "  # Convert existing COCO dataset (all categories)"
    echo "  $0 convert annotations.json output.jsonl"
    echo ""
    echo "  # Convert with category filter"
    echo "  $0 convert annotations.json output.jsonl pipe screw bolt"
    echo ""
    echo "  # Analyze converted data"
    echo "  $0 analyze output.jsonl"
    exit 1
    ;;
esac

echo ""
echo "Done!"
