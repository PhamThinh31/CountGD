#!/bin/bash

set -e

PHASE=$1
IMAGE_DIR=${2:-"data/test_images"}
GT_FILE=${3:-"data/ground_truth.csv"}
SAMPLE_IMAGE=${4:-"data/test_images/sample.jpg"}
TEXT_PROMPT=${5:-"pipe"}

case $PHASE in

  "setup")
    echo "=== PHASE 0: SETUP ==="
    echo "Installing dependencies..."
    pip install -r requirements_profiling.txt

    echo "Downloading SAM variants..."
    ./download_sam_variants.sh

    echo "Checking GPU..."
    python gpu_benchmark.py

    echo "Setup complete!"
    ;;

  "profile")
    echo "=== PHASE 1: PROFILING BASELINE ==="
    echo "Running profiling on H100/3090..."

    python profiler.py \
      --image_path "$SAMPLE_IMAGE" \
      --text "$TEXT_PROMPT" \
      --output_json profiling_results/baseline_no_sam.json \
      --num_runs 10

    python profiler.py \
      --image_path "$SAMPLE_IMAGE" \
      --text "$TEXT_PROMPT" \
      --sam_tt_norm \
      --sam_model_type vit_h \
      --output_json profiling_results/full_sam_vit_h.json \
      --num_runs 10

    echo "Profiling complete! Check profiling_results/"
    ;;

  "batch_profile")
    echo "=== PHASE 1b: BATCH PROFILING (All SAM Variants) ==="

    python batch_profiler.py \
      --image_path "$SAMPLE_IMAGE" \
      --text "$TEXT_PROMPT" \
      --output_dir profiling_results \
      --num_runs 10

    echo "Batch profiling complete! Check profiling_results/"
    ;;

  "optimize")
    echo "=== PHASE 2: APPLY OPTIMIZATIONS ==="

    echo "Testing SAM ViT-B (lightweight)..."
    python profiler.py \
      --image_path "$SAMPLE_IMAGE" \
      --text "$TEXT_PROMPT" \
      --sam_tt_norm \
      --sam_model_type vit_b \
      --sam_model_path checkpoints/sam_vit_b_01ec64.pth \
      --output_json profiling_results/sam_vit_b.json \
      --num_runs 10

    echo "Testing INT8 quantization..."
    python optimization_phase2.py \
      --optimization int8 \
      --output_path checkpoints/model_int8.pth \
      --benchmark \
      --num_runs 100

    echo "Testing FP16..."
    python optimization_phase2.py \
      --optimization fp16 \
      --output_path checkpoints/model_fp16.pth \
      --benchmark \
      --num_runs 100

    echo "Testing early exit..."
    python early_exit_inference.py \
      --image_path "$SAMPLE_IMAGE" \
      --text "$TEXT_PROMPT" \
      --sam_tt_norm \
      --enable_early_exit \
      --early_exit_confidence 0.95 \
      --max_sam_iterations 3

    echo "Optimizations complete!"
    ;;

  "accuracy")
    echo "=== PHASE 3: ACCURACY BENCHMARK ==="

    if [ ! -d "$IMAGE_DIR" ]; then
      echo "Error: Image directory $IMAGE_DIR not found"
      exit 1
    fi

    if [ ! -f "$GT_FILE" ]; then
      echo "Error: Ground truth file $GT_FILE not found"
      exit 1
    fi

    echo "Running accuracy tests..."

    echo "Baseline (no SAM)..."
    python accuracy_benchmark.py \
      --image_dir "$IMAGE_DIR" \
      --ground_truth "$GT_FILE" \
      --text_prompt "$TEXT_PROMPT" \
      --output_json accuracy_results/baseline_no_sam.json \
      --save_predictions accuracy_results/predictions_baseline.csv

    echo "Full SAM ViT-H..."
    python accuracy_benchmark.py \
      --image_dir "$IMAGE_DIR" \
      --ground_truth "$GT_FILE" \
      --text_prompt "$TEXT_PROMPT" \
      --sam_tt_norm \
      --sam_model_type vit_h \
      --output_json accuracy_results/full_sam_vit_h.json \
      --save_predictions accuracy_results/predictions_full_sam.csv

    echo "Optimized SAM ViT-B..."
    python accuracy_benchmark.py \
      --image_dir "$IMAGE_DIR" \
      --ground_truth "$GT_FILE" \
      --text_prompt "$TEXT_PROMPT" \
      --sam_tt_norm \
      --sam_model_type vit_b \
      --sam_model_path checkpoints/sam_vit_b_01ec64.pth \
      --output_json accuracy_results/sam_vit_b.json \
      --save_predictions accuracy_results/predictions_sam_vit_b.csv

    echo "Accuracy tests complete! Check accuracy_results/"
    ;;

  "compare")
    echo "=== PHASE 4: COMPREHENSIVE COMPARISON ==="

    python compare_optimizations.py \
      --image_dir "$IMAGE_DIR" \
      --ground_truth "$GT_FILE" \
      --sample_image "$SAMPLE_IMAGE" \
      --text "$TEXT_PROMPT" \
      --output_dir final_comparison

    echo "Comparison complete! Check final_comparison/"
    echo "Key files:"
    echo "  - final_comparison/full_comparison.csv"
    echo "  - final_comparison/comprehensive_comparison.png"
    ;;

  "all")
    echo "=== RUNNING ALL PHASES ==="

    bash $0 setup
    bash $0 profile "$IMAGE_DIR" "$GT_FILE" "$SAMPLE_IMAGE" "$TEXT_PROMPT"
    bash $0 optimize "$IMAGE_DIR" "$GT_FILE" "$SAMPLE_IMAGE" "$TEXT_PROMPT"
    bash $0 accuracy "$IMAGE_DIR" "$GT_FILE" "$SAMPLE_IMAGE" "$TEXT_PROMPT"
    bash $0 compare "$IMAGE_DIR" "$GT_FILE" "$SAMPLE_IMAGE" "$TEXT_PROMPT"

    echo "=== ALL PHASES COMPLETE ==="
    ;;

  *)
    echo "Usage: $0 {setup|profile|batch_profile|optimize|accuracy|compare|all} [IMAGE_DIR] [GT_FILE] [SAMPLE_IMAGE] [TEXT_PROMPT]"
    echo ""
    echo "Phases:"
    echo "  setup          - Install dependencies, download models, check GPU"
    echo "  profile        - Profile baseline and full SAM (Phase 1)"
    echo "  batch_profile  - Profile all SAM variants in parallel"
    echo "  optimize       - Test optimizations: SAM-ViT-B, INT8, FP16, early exit (Phase 2)"
    echo "  accuracy       - Run accuracy benchmarks on dataset (Phase 3)"
    echo "  compare        - Generate comprehensive comparison report (Phase 4)"
    echo "  all            - Run all phases sequentially"
    echo ""
    echo "Example:"
    echo "  $0 setup"
    echo "  $0 profile data/images data/gt.csv data/images/sample.jpg 'blue pipe'"
    echo "  $0 all data/images data/gt.csv data/images/sample.jpg pipe"
    exit 1
    ;;
esac

echo ""
echo "Phase '$PHASE' completed successfully!"
