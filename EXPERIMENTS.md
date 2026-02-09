# CountGD Experiments Log

## Baseline Reference

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE | Source |
|-------|---------|----------|----------|-----------|--------|
| **CountGD (Paper)** | **7.09** | **26.05** | **5.74** | **24.09** | Official checkpoint |

---

## Experiment Results Summary

### Verified Baseline

| Experiment | Checkpoint | Val MAE | Val RMSE | Notes |
|------------|------------|---------|----------|-------|
| Baseline eval | `checkpoint_fsc147_best.pth` | 7.09 | 26.05 | Reproduced paper result |
| Baseline eval (w/ flags) | `checkpoint_fsc147_best.pth` | 7.08 | 26.00 | With `--crop --sam_tt_norm --remove_bad_exemplar` |

---

### Upgrade Experiments

#### 1. Density Head Experiments

| Experiment | Starting Checkpoint | Config | Best Val MAE | Best Val RMSE | Epochs | Notes |
|------------|---------------------|--------|--------------|---------------|--------|-------|
| `density_head` | GroundingDINO | Swin-B frozen, density_coef=0.5, sigma=3.0, no GIoU | 8.45 | 44.52 | 30 | Density loss collapsed to 0 |
| `density_fixed_v2` | GroundingDINO | Swin-B frozen, density_coef=1.0, sigma=8.0, cosine LR | 11.80 | 50.58 | 50 | Fixed density loss (Smooth L1 + normalization) |
| `density_fixed_v2_with_unfreeze_backbone` | GroundingDINO | Swin-B unfrozen, density_coef=1.0, sigma=8.0 | 10.23 | 46.33 | 50 | Unfrozen backbone |
| `density_finetune` | CountGD best | Swin-B frozen, density_coef=0.1, sigma=8.0, lr=1e-5 | **8.55** | 43.29 | 20 | **Best density result** |
| `density_finetune_unfrezee` | CountGD best | Swin-B unfrozen, density_coef=0.1, sigma=8.0, lr=1e-5 | 8.63 | 43.22 | 20 | Unfreezing didn't help |
| `density_finetune_tiny_coef` | CountGD best | density_coef=0.01, sigma=8.0, lr=5e-6 | TBD | TBD | 30 | Running |
| `upgrade5_density_head_with_countgd_best` | CountGD best | density_coef=0.5, sigma=3.0 | 10.31 | 46.13 | 27 | Original density (before fix) |

#### 2. Swin-V2-B Backbone Experiments

| Experiment | Starting Checkpoint | Config | Best Val MAE | Best Val RMSE | Epochs | Notes |
|------------|---------------------|--------|--------------|---------------|--------|-------|
| `training_upgrade_5` | GroundingDINO | Swin-V2-B frozen, GIoU=0.2, cosine LR | 23.75 | 77.06 | 50 | Backbone frozen (0 trainable params) - BAD |
| `upgrade5_swinv2_and_density_head_without_giou` | GroundingDINO | Swin-V2-B frozen + density, no GIoU | 24.50 | 76.30 | 26 | Incomplete training |
| `swinv2_trainable` | GroundingDINO | Swin-V2-B unfrozen, lr=5e-5, lr_backbone=1e-5 | 19.88 | 70.98 | 17+ | Training (needs more epochs) |

#### 3. Other Experiments

| Experiment | Starting Checkpoint | Config | Best Val MAE | Best Val RMSE | Notes |
|------------|---------------------|--------|--------------|---------------|-------|
| `baseline_finetune_no_density` | CountGD best | No density, lr=1e-5, cosine LR | TBD | TBD | Running |
| `density_finetune_with_giou` | CountGD best | density + GIoU=2.0, lr=5e-6 | TBD | TBD | Planned |

---

### Multi-Scale TTA Experiment

| Experiment | Scales | NMS Threshold | Val MAE | Val RMSE | Time/img | Notes |
|------------|--------|---------------|---------|----------|----------|-------|
| Normal eval | 1.0 | - | 7.09 | 26.05 | 0.58s | Baseline |
| TTA | 0.75, 1.0, 1.25 | 0.5 | 13.06 | 44.25 | 0.40s | **HURT performance** |

**TTA Analysis**: TTA made results worse because:
1. NMS threshold (0.5) too aggressive - removes valid detections
2. Speed anomaly (faster than baseline) suggests implementation issue
3. Normal path uses NO NMS for counting - just counts all boxes above threshold

---

## Training Commands Reference

### Density Head (Fixed Version)
```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --output_dir ./output/density_fixed_v2 \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            num_queries=900 num_select=900 \
            use_density_head=True density_loss_coef=1.0 density_sigma=8.0 \
            epochs=50 cosine_lr=True
```

### Density Finetune from CountGD Best (Best Result)
```bash
CUDA_VISIBLE_DEVICES=2 python -u main.py \
  --output_dir ./output/density_finetune \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path /path/to/CountGD/checkpoints/checkpoint_fsc147_best.pth \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            num_queries=900 num_select=900 \
            use_density_head=True density_loss_coef=0.1 density_sigma=8.0 \
            lr=1e-5 epochs=20 cosine_lr=True
```

### Swin-V2-B Trainable
```bash
CUDA_VISIBLE_DEVICES=5 python -u main.py \
  --output_dir ./output/swinv2_trainable \
  -c config/cfg_fsc147_vit_b_swinv2.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --finetune_ignore backbone.0 feature_map_proj input_proj \
  --amp \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            backbone=swinv2_base_window12to24_192to384.ms_in22k_ft_in1k \
            num_queries=900 num_select=900 \
            batch_size=4 \
            lr=5e-5 lr_backbone=1e-5 \
            epochs=100 cosine_lr=True cosine_warmup_epochs=5
```

### Evaluation
```bash
python main_inference.py \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path /path/to/checkpoint.pth \
  --eval \
  --options text_encoder_type=checkpoints/bert-base-uncased num_queries=900 num_select=900
```

---

## Key Findings

### What Works
1. **Starting from CountGD best checkpoint** - Already well-optimized
2. **Lower learning rate (1e-5 to 5e-6)** - Prevents destroying learned features
3. **Frozen backbone** - Unfreezing Swin-B doesn't help when starting from CountGD best

### What Doesn't Work
1. **Density head** - Adds overhead but doesn't improve MAE (8.55 vs 7.09)
2. **Swin-V2-B replacement** - Needs extensive training, currently at ~20 MAE
3. **Multi-scale TTA** - Hurts performance with current implementation
4. **High learning rates** - Destroys pretrained features

### Issues Found & Fixed
1. **Density loss collapse** - Fixed by normalizing targets + using Smooth L1 loss
2. **Swin-V2 frozen with random weights** - Fixed by removing `backbone.0` from `freeze_keywords`
3. **LR scheduler mismatch** - Fixed by catching incompatible state dict errors
4. **NHWC vs NCHW** - Fixed by permuting Swin-V2 outputs in TimmBackbone

---

## FSC-147 SOTA Comparison

| Rank | Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|------|-------|---------|----------|----------|-----------|
| 1 | **CountGD** | **7.10** | **26.08** | **5.74** | **24.09** |
| 2 | GeCo | 9.52 | 43.00 | 7.91 | 54.28 |
| 3 | CACViT | 10.63 | 37.95 | 9.13 | 48.96 |
| 4 | SSD | 9.73 | 29.72 | 9.58 | 64.13 |
| 5 | LOCA | 10.24 | 32.56 | 10.79 | 56.97 |
| 6 | CounTR | 13.13 | 49.83 | 11.95 | 91.23 |

**CountGD is already #1 on FSC-147.** Beating 7.09 Val MAE is extremely challenging.

---

## Thesis Directions

Since CountGD is SOTA, alternative thesis contributions:

1. **Failure Analysis** - When/why does CountGD fail? Per-category analysis
2. **Efficiency** - Make CountGD faster (distillation, pruning)
3. **Generalization** - Test on other datasets (CARPK, ShanghaiTech)
4. **Domain Adaptation** - Fine-tune for specific domains (medical, agriculture)
5. **Hybrid Detection+Density** - Use density for count estimation, detection for localization

---

## Files Modified

| File | Changes |
|------|---------|
| `models/GroundingDINO/backbone/timm_backbone.py` | New Swin-V2-B wrapper with dynamic resolution support |
| `models/GroundingDINO/density_head.py` | New density regression head |
| `models/GroundingDINO/groundingdino.py` | Density head integration, loss computation |
| `models/GroundingDINO/backbone/backbone.py` | Routing for timm backbones |
| `config/cfg_fsc147_vit_b.py` | Added upgrade config options |
| `config/cfg_fsc147_vit_b_swinv2.py` | Swin-V2-B specific config |
| `main.py` | Cosine LR, torch.compile, LR scheduler fix |
| `main_inference.py` | TTA implementation |
| `engine_inference.py` | TTA forward function |

All changes also applied to `models_inference/` directory for inference compatibility.
