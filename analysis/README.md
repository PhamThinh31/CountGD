# CountGD Analysis Framework

## Thesis: "Understanding and Improving Detection-Based Object Counting: An Analysis of CountGD with Density-Guided Enhancement"

This directory contains analysis scripts and tools for comprehensive evaluation of CountGD.

## Analysis Components

### C.1: Comprehensive Ablation Study
- **Per-category analysis**: Evaluate MAE/RMSE per object category
- **Count-stratified evaluation**: Performance on low/medium/high count images
- **Failure case identification**: Systematic error analysis with visualizations
- **Error pattern analysis**: Overcounting vs undercounting patterns

### D.1: Density-Guided Hybrid Approach
- **Detection + Density ensemble**: Combine both signals for robust counting
- **Confidence-weighted fusion**: Weight by prediction confidence
- **Adaptive fusion**: Use density for high-count, detection for low-count
- **Count consistency**: Compare detection count with density integral

## Scripts

| Script | Description |
|--------|-------------|
| `per_category_eval.py` | Evaluate model per object category |
| `count_stratified_eval.py` | Evaluate by count ranges (low/med/high/very_high) |
| `failure_analysis.py` | Identify and visualize failure cases |
| `density_hybrid.py` | Density-guided hybrid counting evaluation |
| `run_all_analysis.py` | Run all analyses and generate report |

## Usage

### Run Full Analysis Pipeline
```bash
python analysis/run_all_analysis.py \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path /path/to/checkpoint_fsc147_best.pth \
  --output_dir ./analysis_results \
  --options text_encoder_type=checkpoints/bert-base-uncased num_queries=900 num_select=900
```

### Individual Analyses
```bash
# Per-category evaluation
python analysis/per_category_eval.py \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path /path/to/checkpoint.pth \
  --output_dir ./analysis_results \
  --options text_encoder_type=checkpoints/bert-base-uncased num_queries=900 num_select=900

# Count-stratified evaluation
python analysis/count_stratified_eval.py \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path /path/to/checkpoint.pth \
  --output_dir ./analysis_results \
  --options text_encoder_type=checkpoints/bert-base-uncased num_queries=900 num_select=900

# Failure analysis with visualizations
python analysis/failure_analysis.py \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path /path/to/checkpoint.pth \
  --output_dir ./analysis_results \
  --top_k 50 \
  --visualize \
  --options text_encoder_type=checkpoints/bert-base-uncased num_queries=900 num_select=900

# Density hybrid evaluation (requires model trained with density head)
python analysis/density_hybrid.py \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path /path/to/checkpoint_with_density.pth \
  --output_dir ./analysis_results \
  --options text_encoder_type=checkpoints/bert-base-uncased num_queries=900 num_select=900 use_density_head=True
```

## Output Files

Results are saved to `analysis_results/`:

| File | Description |
|------|-------------|
| `per_category_results.csv` | MAE/RMSE/bias per category |
| `count_stratified_results.csv` | Performance by count range |
| `count_stratified_detailed.csv` | Per-image predictions with count range |
| `failure_cases.json` | Top K failure cases with metadata |
| `failure_patterns.json` | Aggregated failure pattern analysis |
| `all_predictions.csv` | All predictions for every image |
| `hybrid_methods_results.csv` | Comparison of hybrid counting methods |
| `hybrid_predictions_detailed.csv` | Per-image predictions for all methods |
| `analysis_report.txt` | Comprehensive summary report |
| `failure_visualizations/` | Visualization images of failure cases |

## Thesis Structure

The analysis supports the following thesis contributions:

1. **Comprehensive Analysis (C.1)**
   - Which object categories are hardest/easiest for CountGD?
   - How does performance degrade with increasing object count?
   - What are the systematic failure patterns?

2. **Density-Guided Enhancement (D.1)**
   - Can combining detection and density improve counting?
   - Which fusion strategy works best?
   - When does density help vs. hurt?

3. **Practical Recommendations**
   - Guidelines for deploying CountGD in production
   - When to use text vs. exemplar prompts
   - Handling high-count scenarios
