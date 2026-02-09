"""
Per-Category Evaluation for CountGD

Analyzes model performance broken down by object category to identify
which categories are easy/hard for the model.

Output: per_category_results.csv with MAE, RMSE, count per category
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main_inference import get_args_parser
from util.slconfig import SLConfig


def load_model_and_data(args):
    """Load model and dataset."""
    from datasets import build_dataset
    from models import build_model
    from util.misc import clean_state_dict

    # Build model
    model, criterion, postprocessors = build_model(args)
    model.to(args.device)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(clean_state_dict(checkpoint), strict=False)

    # Build dataset
    dataset_val = build_dataset(image_set='val', args=args)

    return model, dataset_val, postprocessors


def get_category_name(dataset, target):
    """Extract category name from target."""
    if 'category_name' in target:
        return target['category_name']
    elif 'labels' in target and hasattr(dataset, 'cat_list'):
        label_idx = target['labels'][0].item()
        if label_idx < len(dataset.cat_list):
            return dataset.cat_list[label_idx]
    return 'unknown'


def run_inference(model, sample, exemplars, caption, args, device):
    """Run model inference on a single sample."""
    from util.misc import nested_tensor_from_tensor_list

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(
                nested_tensor_from_tensor_list([sample.to(device)]),
                [exemplars.to(device)],
                [torch.tensor([0]).to(device)],
                captions=[caption],
            )
    return outputs


def count_predictions(outputs, args, tokenized_caption):
    """Count predictions above threshold."""
    logits = outputs['pred_logits'][0].sigmoid()

    # Find end_idx for text token filtering
    end_idx = 1
    for token_ind in range(len(tokenized_caption)):
        if tokenized_caption[token_ind] == 1012:  # Period token
            end_idx = token_ind
            break

    # Apply thresholds
    box_mask = logits.max(dim=-1).values > args.box_threshold
    logits = logits[box_mask]

    if logits.shape[0] > 0:
        text_mask = (logits[:, 1:end_idx] > args.text_threshold).sum(dim=-1) == (end_idx - 1)
        pred_count = text_mask.sum().item()
    else:
        pred_count = 0

    return pred_count


def evaluate_per_category(model, dataset, args, device):
    """Evaluate model performance per category."""
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_type)

    # Results storage
    category_results = defaultdict(lambda: {'errors': [], 'gt_counts': [], 'pred_counts': []})

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating {len(dataset)} images...")

    for idx, (samples, targets) in enumerate(tqdm(dataloader)):
        target = {k: v[0] if isinstance(v, list) else v.squeeze(0) for k, v in targets[0].items()}
        sample = samples.tensors[0]

        # Get category
        category = get_category_name(dataset, target)

        # Get exemplars
        if 'exemplars' in target:
            exemplars = target['exemplars']
        else:
            exemplars = target.get('boxes', torch.zeros(0, 4))[:3]

        # Build caption
        caption = f"{category} ."
        tokenized = tokenizer(caption, return_tensors='pt')

        # Run inference
        outputs = run_inference(model, sample, exemplars, caption, args, device)

        # Count predictions
        pred_count = count_predictions(outputs, args, tokenized['input_ids'][0])

        # Get ground truth count
        if 'labels_uncropped' in target:
            gt_count = target['labels_uncropped'].shape[0]
        else:
            gt_count = target['boxes'].shape[0]

        # Store results
        error = abs(gt_count - pred_count)
        category_results[category]['errors'].append(error)
        category_results[category]['gt_counts'].append(gt_count)
        category_results[category]['pred_counts'].append(pred_count)

    return category_results


def compute_category_metrics(category_results):
    """Compute MAE, RMSE, and other metrics per category."""
    results = []

    for category, data in category_results.items():
        errors = np.array(data['errors'])
        gt_counts = np.array(data['gt_counts'])
        pred_counts = np.array(data['pred_counts'])

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        n_samples = len(errors)
        avg_gt_count = np.mean(gt_counts)
        avg_pred_count = np.mean(pred_counts)

        # Relative MAE (normalized by average count)
        rel_mae = mae / avg_gt_count if avg_gt_count > 0 else 0

        # Count bias (positive = overcounting, negative = undercounting)
        bias = np.mean(pred_counts - gt_counts)

        results.append({
            'category': category,
            'n_samples': n_samples,
            'mae': mae,
            'rmse': rmse,
            'rel_mae': rel_mae,
            'avg_gt_count': avg_gt_count,
            'avg_pred_count': avg_pred_count,
            'bias': bias,
            'min_error': np.min(errors),
            'max_error': np.max(errors),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser('Per-Category Evaluation', parents=[get_args_parser()])
    parser.add_argument('--output_dir', default='./analysis_results', help='Output directory')
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    cfg = SLConfig.fromfile(args.config_file)
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)
    args = cfg
    args.device = device

    # Load model and data
    print("Loading model and data...")
    model, dataset, postprocessors = load_model_and_data(args)

    # Evaluate
    print("Running per-category evaluation...")
    category_results = evaluate_per_category(model, dataset, args, device)

    # Compute metrics
    df = compute_category_metrics(category_results)
    df = df.sort_values('mae', ascending=False)

    # Save results
    output_path = os.path.join(args.output_dir, 'per_category_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("PER-CATEGORY EVALUATION RESULTS")
    print("="*80)
    print(f"\nTotal categories: {len(df)}")
    print(f"Overall MAE: {df['mae'].mean():.2f}")
    print(f"Overall RMSE: {np.sqrt((df['rmse']**2).mean()):.2f}")

    print("\n--- TOP 10 HARDEST CATEGORIES (by MAE) ---")
    print(df.head(10)[['category', 'n_samples', 'mae', 'rmse', 'avg_gt_count', 'bias']].to_string(index=False))

    print("\n--- TOP 10 EASIEST CATEGORIES (by MAE) ---")
    print(df.tail(10)[['category', 'n_samples', 'mae', 'rmse', 'avg_gt_count', 'bias']].to_string(index=False))

    print("\n--- BIAS ANALYSIS ---")
    overcounting = df[df['bias'] > 1]
    undercounting = df[df['bias'] < -1]
    print(f"Categories with overcounting bias (>1): {len(overcounting)}")
    print(f"Categories with undercounting bias (<-1): {len(undercounting)}")


if __name__ == '__main__':
    main()
