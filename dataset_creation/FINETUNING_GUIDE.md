# Fine-tuning CountGD on Construction Dataset

This guide explains how to use a pre-trained FSC-147 CountGD model and fine-tune it on your custom construction dataset.

## Overview

CountGD is trained on FSC-147 (a general object counting dataset). To apply it to construction materials (scaffolding, rebar, pipes, etc.), you'll fine-tune the pre-trained model while preserving its general counting knowledge.

## Quick Start

```bash
# 1. Download pre-trained FSC-147 model (manual step)
# Download from: https://drive.google.com/file/d/1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI/view
# Save to: checkpoints/checkpoint_fsc147_best.pth

# 2. Prepare your construction dataset (if not done already)
python download_and_convert.py \
    --config dataset_configs_latest.json \
    --api_key YOUR_ROBOFLOW_API_KEY \
    --output_dir construction_data

# 3. Run fine-tuning
bash train_construction.sh
```

## Detailed Steps

### 1. Pre-trained Model Setup

**Required Downloads:**

```bash
# a) Swin-B backbone (GroundingDINO)
mkdir -p checkpoints
wget -P checkpoints https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# b) BERT text encoder
mkdir -p checkpoints/bert-base-uncased
cd checkpoints/bert-base-uncased
wget https://huggingface.co/bert-base-uncased/resolve/main/config.json
wget https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
cd ../..

# c) FSC-147 pre-trained CountGD model (MANUAL DOWNLOAD)
# Download: https://drive.google.com/file/d/1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI/view
# Save as: checkpoints/checkpoint_fsc147_best.pth
```

### 2. Dataset Preparation

Your construction dataset should be in the format created by `download_and_convert.py`:

```
construction_data/final/
├── images/              # All images
├── train.jsonl          # Training annotations (ODVG format)
├── val.jsonl            # Validation annotations (ODVG format)
├── test.jsonl           # Test annotations (ODVG format)
├── val_coco.json        # Validation in COCO format
├── test_coco.json       # Test in COCO format
├── label_map.json       # Category mapping
└── dataset_config.json  # Dataset configuration
```

**Dataset config structure (`dataset_config.json`):**
```json
{
  "train": [{
    "root": "/absolute/path/to/images",
    "anno": "/absolute/path/to/train.jsonl",
    "label_map": "/absolute/path/to/label_map.json",
    "dataset_mode": "odvg"
  }],
  "val": [{
    "root": "/absolute/path/to/images",
    "anno": "/absolute/path/to/val_coco.json",
    "label_map": null,
    "dataset_mode": "coco"
  }]
}
```

### 3. Fine-tuning Configuration

**Key settings in `config/cfg_construction_finetune.py`:**

```python
# Freeze pre-trained layers to preserve FSC-147 knowledge
freeze_keywords = ['backbone.0', 'bert']  # Freeze visual and text encoders

# Learning rates
lr = 0.0001              # Base learning rate
lr_backbone = 1e-05      # Lower LR for backbone (if unfrozen)

# Training schedule
epochs = 10              # Fewer epochs for fine-tuning
lr_drop = 7              # Drop LR at epoch 7

# Your construction categories
label_list = [
    'Scaffolding Frame',
    'Scaffolding Crossbar',
    'Scaffolding Jack Base',
    'Scaffolding Steel Pipe',
    'Vertical Post',
    'Steel Rebar'
]
```

### 4. Run Training

**Basic fine-tuning command:**
```bash
python -u main.py \
    --config_file config/cfg_construction_finetune.py \
    --datasets construction_data/final/dataset_config.json \
    --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth \
    --output_dir output/construction_finetune \
    --options text_encoder_type=checkpoints/bert-base-uncased
```

**Resume from checkpoint:**
```bash
python -u main.py \
    --config_file config/cfg_construction_finetune.py \
    --datasets construction_data/final/dataset_config.json \
    --resume output/construction_finetune/checkpoint_best.pth \
    --output_dir output/construction_finetune \
    --options text_encoder_type=checkpoints/bert-base-uncased
```

### 5. Monitor Training

**Check logs:**
```bash
tail -f output/construction_finetune/log.txt
```

**Checkpoints saved:**
- `checkpoint_best.pth` - Best model based on validation performance
- `checkpoint_latest.pth` - Most recent checkpoint
- `checkpoint_0009.pth` - Checkpoint at epoch 9 (saved every 5 epochs)

### 6. Test Fine-tuned Model

**Single image inference:**
```bash
python single_image_inference.py \
    --checkpoint output/construction_finetune/checkpoint_best.pth \
    --image path/to/test/image.jpg \
    --text_prompt "Scaffolding Frame"
```

**Batch inference:**
```bash
python main_inference.py \
    --checkpoint output/construction_finetune/checkpoint_best.pth \
    --config_file config/cfg_construction_finetune.py \
    --datasets construction_data/final/dataset_config.json
```

## Transfer Learning Strategy

### What Gets Frozen?

**Frozen layers** (`freeze_keywords = ['backbone.0', 'bert']`):
- `backbone.0` - Swin-B visual encoder (learns image features)
- `bert` - BERT text encoder (understands text prompts)

These are frozen to preserve the general counting knowledge from FSC-147.

**Fine-tuned layers:**
- Detection heads (bounding box regression, classification)
- Fusion layers (combining visual and text features)
- Query embeddings (object queries for transformer decoder)

### Why This Works?

1. **Visual features** learned from FSC-147 (cars, people, objects) transfer well to construction objects
2. **Text understanding** from BERT helps with new category names ("Scaffolding Frame")
3. **Detection heads** adapt to construction-specific object shapes and sizes
4. **Faster convergence** - only 10 epochs needed vs. 30 from scratch

## Advanced Options

### Adjust Freezing Strategy

**To unfreeze everything (full fine-tuning):**
```python
# In config/cfg_construction_finetune.py
freeze_keywords = []  # Unfreeze all layers
```

**To freeze only the backbone:**
```python
freeze_keywords = ['backbone.0']  # Keep BERT trainable
```

### Hyperparameter Tuning

**For smaller datasets (<500 images):**
```python
epochs = 15              # More epochs
lr = 5e-05               # Lower learning rate
batch_size = 2           # Smaller batch size
```

**For larger datasets (>2000 images):**
```python
epochs = 10              # Fewer epochs
lr = 0.0001              # Standard learning rate
batch_size = 8           # Larger batch size (if GPU allows)
```

### Multi-GPU Training

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env main.py \
    --config_file config/cfg_construction_finetune.py \
    --datasets construction_data/final/dataset_config.json \
    --pretrain_model_path checkpoints/checkpoint_fsc147_best.pth \
    --output_dir output/construction_finetune
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `config/cfg_construction_finetune.py` | Training configuration |
| `train_construction.sh` | Training script |
| `main.py` | Main training entry point |
| `engine.py` | Training loop implementation |
| `checkpoints/checkpoint_fsc147_best.pth` | Pre-trained FSC-147 model |
| `construction_data/final/dataset_config.json` | Dataset paths |

## Command-line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--config_file` | Training config file | `config/cfg_construction_finetune.py` |
| `--datasets` | Dataset config JSON | `construction_data/final/dataset_config.json` |
| `--pretrain_model_path` | Pre-trained checkpoint | `checkpoints/checkpoint_fsc147_best.pth` |
| `--resume` | Resume from checkpoint | `output/checkpoint_best.pth` |
| `--output_dir` | Output directory | `output/construction_finetune` |
| `--options` | Override config options | `text_encoder_type=checkpoints/bert-base-uncased` |

## Troubleshooting

### Issue: "KeyError: 'model'" when loading checkpoint

**Solution:** Make sure you're using `--pretrain_model_path` (not `--resume`) for the FSC-147 checkpoint:
```bash
--pretrain_model_path checkpoints/checkpoint_fsc147_best.pth  # Correct
# NOT --resume checkpoints/checkpoint_fsc147_best.pth
```

### Issue: Out of memory (OOM)

**Solution:** Reduce batch size in config:
```python
batch_size = 2  # Reduce from 4
```

### Issue: Model not learning / loss not decreasing

**Solutions:**
1. Check if too many layers are frozen - try unfreezing BERT:
   ```python
   freeze_keywords = ['backbone.0']  # Only freeze visual encoder
   ```
2. Increase learning rate:
   ```python
   lr = 0.0002  # Increase from 0.0001
   ```

### Issue: Categories mismatch

**Solution:** Ensure `label_list` in config matches your dataset categories:
```bash
# Check your categories
cat construction_data/final/label_map.json

# Update config/cfg_construction_finetune.py accordingly
label_list = ['Scaffolding Frame', 'Steel Rebar', ...]
```

## Next Steps: Optimization for Edge Devices

After fine-tuning, see your thesis workflow for optimization:
1. **Model distillation** - Replace SAM-ViT-H with SAM-ViT-B
2. **Quantization** - FP32 → FP16 or INT8
3. **Early exit** - Stop SAM processing early for simple cases
4. **Pruning** - Remove less important weights

See `optimization_phase2.py` for implementation details.
