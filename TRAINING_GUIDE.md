# CountGD Training Guide (with 7 Upgrades)

## Prerequisites

### 1. Copy checkpoints from the downloaded training code

```bash
# Copy pretrained weights + BERT model
cp -r "/Users/thinhpham/Downloads/reproduced_countgd_results - compressed/exemp_and_text_fus-823d52afa87fb4b7b6e46d666e211252a84f0253/checkpoints" .
```

After this you should have:
```
checkpoints/
  groundingdino_swinb_cogcoor.pth     # 895 MB — pretrained GroundingDINO Swin-B
  bert-base-uncased/                   # BERT tokenizer + model weights
```

### 2. Download FSC-147 dataset images

Download from: https://github.com/cvlab-stonybrook/LearningToCountEverything

The dataset JSON expects images at: `../CountGD/FSC147_384_V2/images_384_VarV2`

If your images are elsewhere, edit `config/datasets_fsc147.json` to point to the correct path.

### 3. Install dependencies

```bash
pip install -r requirements.txt
# Build the deformable attention CUDA op
cd models/GroundingDINO/csrc
python setup.py build install
cd ../../..
```

---

## Training Commands

### Baseline (original CountGD, no upgrades)

```bash
python main.py \
  --output_dir ./output/baseline \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            set_cost_giou=0.0 giou_loss_coef=0.0 \
            num_queries=900 num_select=900 \
            cosine_lr=False use_density_head=False
```

### Upgrade 1 only: GIoU Loss

```bash
python main.py \
  --output_dir ./output/upgrade1_giou \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            set_cost_giou=2.0 giou_loss_coef=2.0 \
            num_queries=900 num_select=900
```

### Upgrade 2 only: Reduced Queries (400)

```bash
python main.py \
  --output_dir ./output/upgrade2_queries400 \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --finetune_ignore tgt_embed refpoint_embed \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            set_cost_giou=0.0 giou_loss_coef=0.0 \
            num_queries=400 num_select=400
```

### Upgrade 3 only: torch.compile

```bash
python main.py \
  --output_dir ./output/upgrade3_compile \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --use_torch_compile \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            set_cost_giou=0.0 giou_loss_coef=0.0 \
            num_queries=900 num_select=900
```

### Upgrade 4 only: Cosine LR + Longer Training

```bash
python main.py \
  --output_dir ./output/upgrade4_cosine \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            set_cost_giou=0.0 giou_loss_coef=0.0 \
            num_queries=900 num_select=900 \
            cosine_lr=True cosine_warmup_epochs=2 cosine_lr_min=1e-6 epochs=50
```

### Upgrade 5 only: Swin-V2-B Backbone

```bash
python main.py \
  --output_dir ./output/upgrade5_swinv2 \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --finetune_ignore backbone.0 feature_map_proj input_proj \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            set_cost_giou=0.0 giou_loss_coef=0.0 \
            num_queries=900 num_select=900 \
            backbone=swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
```

Alternative backbones (same command, just swap the backbone= value):
- `backbone=convnextv2_base.fcmae_ft_in22k_in1k_384`   (ConvNeXt-V2-B)
- `backbone=focalnet_base_lrf.in1k`                      (FocalNet-B)

### Upgrade 6 only: Density Map Head

```bash
python main.py \
  --output_dir ./output/upgrade6_density \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --find_unused_params \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            set_cost_giou=0.0 giou_loss_coef=0.0 \
            num_queries=900 num_select=900 \
            use_density_head=True density_loss_coef=0.5 density_sigma=3.0
```

### Upgrade 7: Multi-scale TTA (inference only, no training needed)

```bash
python main_inference.py \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path ./output/baseline/checkpoint_best_regular.pth \
  --eval \
  --use_multiscale_tta \
  --tta_scales 0.75 1.0 1.25 \
  --options text_encoder_type=checkpoints/bert-base-uncased
```

### All upgrades combined (recommended final run)

```bash
python main.py \
  --output_dir ./output/all_upgrades \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path checkpoints/groundingdino_swinb_cogcoor.pth \
  --finetune_ignore backbone.0 feature_map_proj input_proj \
  --find_unused_params \
  --use_torch_compile \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            set_cost_giou=2.0 giou_loss_coef=2.0 \
            num_queries=400 num_select=400 \
            cosine_lr=True cosine_warmup_epochs=2 epochs=50 \
            use_density_head=True density_loss_coef=0.5 \
            backbone=swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
```

Then evaluate with TTA:
```bash
python main_inference.py \
  -c config/cfg_fsc147_vit_b.py \
  --datasets config/datasets_fsc147.json \
  --pretrain_model_path ./output/all_upgrades/checkpoint_best_regular.pth \
  --eval \
  --use_multiscale_tta \
  --tta_scales 0.75 1.0 1.25 \
  --options text_encoder_type=checkpoints/bert-base-uncased \
            backbone=swinv2_base_window12to24_192to384.ms_in22k_ft_in1k \
            num_queries=400 num_select=400 use_density_head=True
```

---

## Backbone Options Comparison

| Backbone | Params | ImageNet (22k->1k) | VRAM (3090) | timm name |
|----------|--------|---------------------|-------------|-----------|
| Swin-B (original) | 88M | 86.4% | OK | `swin_B_384_22k` |
| Swin-V2-B | 88M | 87.1% | OK | `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` |
| ConvNeXt-V2-B | 89M | 87.3% | OK | `convnextv2_base.fcmae_ft_in22k_in1k_384` |
| FocalNet-B | 89M | 87.3% | OK | `focalnet_base_lrf.in1k` |
| EVA-02-L (NOT recommended) | 304M | 89.6% | OOM on 3090 | `eva02_large_patch14_448` |

When switching backbone with --finetune_ignore:
- `backbone.0` — skip loading old backbone weights (shape mismatch)
- `feature_map_proj` — skip loading exemplar projection (channel count may differ)
- `input_proj` — skip loading feature projection layers (channel count may differ)

---

## Notes

- The `--options` flag overrides values from the config file at runtime
- Training logs go to `<output_dir>/info.txt`
- Best checkpoint is saved as `checkpoint_best_regular.pth`
- The original paper trains for 30 epochs on a single A6000 (~24h)
- On a 3090 with batch_size=4, expect similar training time
- RMSE varies across runs; MAE is stable (as noted by the authors)
