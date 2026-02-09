"""
Count-Stratified Evaluation for CountGD

Analyzes model performance broken down by count ranges:
- Low count: 1-10 objects
- Medium count: 11-50 objects
- High count: 51-100 objects
- Very high count: 100+ objects

This reveals whether the model struggles with dense scenes.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from main_inference import get_args_parser
from util.slconfig import SLConfig


# Count range definitions
COUNT_RANGES = {
    'low': (1, 10),
    'medium': (11, 50),
    'high': (51, 100),
    'very_high': (101, float('inf')),
}


def load_model_and_data(args):
    """Load model and dataset."""
    from datasets import build_dataset
    from models import build_model
    from util.misc import clean_state_dict

    model, criterion, postprocessors = build_model(args)
    model.to(args.device)
    model.eval()

    checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(clean_state_dict(checkpoint), strict=False)

    dataset_val = build_dataset(image_set='val', args=args)

    return model, dataset_val, postprocessors


def get_count_range(count):
    """Determine which count range a value belongs to."""
    for range_name, (low, high) in COUNT_RANGES.items():
        if low <= count <= high:
            return range_name
    return 'very_high'


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

    end_idx = 1
    for token_ind in range(len(tokenized_caption)):
        if tokenized_caption[token_ind] == 1012:
            end_idx = token_ind
            break

    box_mask = logits.max(dim=-1).values > args.box_threshold
    logits = logits[box_mask]

    if logits.shape[0] > 0:
        text_mask = (logits[:, 1:end_idx] > args.text_threshold).sum(dim=-1) == (end_idx - 1)
        pred_count = text_mask.sum().item()
    else:
        pred_count = 0

    return pred_count


def get_category_name(dataset, target):
    """Extract category name from target."""
    if 'category_name' in target:
        return target['category_name']
    elif 'labels' in target and hasattr(dataset, 'cat_list'):
        label_idx = target['labels'][0].item()
        if label_idx < len(dataset.cat_list):
            return dataset.cat_list[label_idx]
    return 'unknown'


def evaluate_count_stratified(model, dataset, args, device):
    """Evaluate model performance stratified by count ranges."""
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_type)

    # Results storage by count range
    range_results = {r: {'errors': [], 'gt_counts': [], 'pred_counts': [], 'categories': []}
                     for r in COUNT_RANGES.keys()}

    # Per-image results for detailed analysis
    image_results = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating {len(dataset)} images...")

    for idx, (samples, targets) in enumerate(tqdm(dataloader)):
        target = {k: v[0] if isinstance(v, list) else v.squeeze(0) for k, v in targets[0].items()}
        sample = samples.tensors[0]

        category = get_category_name(dataset, target)

        if 'exemplars' in target:
            exemplars = target['exemplars']
        else:
            exemplars = target.get('boxes', torch.zeros(0, 4))[:3]

        caption = f"{category} ."
        tokenized = tokenizer(caption, return_tensors='pt')

        outputs = run_inference(model, sample, exemplars, caption, args, device)
        pred_count = count_predictions(outputs, args, tokenized['input_ids'][0])

        if 'labels_uncropped' in target:
            gt_count = target['labels_uncropped'].shape[0]
        else:
            gt_count = target['boxes'].shape[0]

        error = abs(gt_count - pred_count)
        count_range = get_count_range(gt_count)

        range_results[count_range]['errors'].append(error)
        range_results[count_range]['gt_counts'].append(gt_count)
        range_results[count_range]['pred_counts'].append(pred_count)
        range_results[count_range]['categories'].append(category)

        image_results.append({
            'image_idx': idx,
            'category': category,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'error': error,
            'count_range': count_range,
        })

    return range_results, image_results


def compute_range_metrics(range_results):
    """Compute metrics per count range."""
    results = []

    for range_name, data in range_results.items():
        if len(data['errors']) == 0:
            continue

        errors = np.array(data['errors'])
        gt_counts = np.array(data['gt_counts'])
        pred_counts = np.array(data['pred_counts'])

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        n_samples = len(errors)
        avg_gt_count = np.mean(gt_counts)
        avg_pred_count = np.mean(pred_counts)
        rel_mae = mae / avg_gt_count if avg_gt_count > 0 else 0
        bias = np.mean(pred_counts - gt_counts)

        # Percentile analysis
        p50_error = np.percentile(errors, 50)
        p90_error = np.percentile(errors, 90)
        p95_error = np.percentile(errors, 95)

        low, high = COUNT_RANGES[range_name]
        range_str = f"{low}-{int(high) if high != float('inf') else '∞'}"

        results.append({
            'count_range': range_name,
            'range_bounds': range_str,
            'n_samples': n_samples,
            'mae': mae,
            'rmse': rmse,
            'rel_mae': rel_mae,
            'avg_gt_count': avg_gt_count,
            'avg_pred_count': avg_pred_count,
            'bias': bias,
            'median_error': p50_error,
            'p90_error': p90_error,
            'p95_error': p95_error,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser('Count-Stratified Evaluation', parents=[get_args_parser()])
    parser.add_argument('--output_dir', default='./analysis_results', help='Output directory')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = SLConfig.fromfile(args.config_file)
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)
    args = cfg
    args.device = device

    print("Loading model and data...")
    model, dataset, postprocessors = load_model_and_data(args)

    print("Running count-stratified evaluation...")
    range_results, image_results = evaluate_count_stratified(model, dataset, args, device)

    # Compute and save range metrics
    df_ranges = compute_range_metrics(range_results)
    df_ranges.to_csv(os.path.join(args.output_dir, 'count_stratified_results.csv'), index=False)

    # Save detailed image results
    df_images = pd.DataFrame(image_results)
    df_images.to_csv(os.path.join(args.output_dir, 'count_stratified_detailed.csv'), index=False)

    # Print summary
    print("\n" + "="*80)
    print("COUNT-STRATIFIED EVALUATION RESULTS")
    print("="*80)

    print("\n--- PERFORMANCE BY COUNT RANGE ---")
    print(df_ranges[['count_range', 'range_bounds', 'n_samples', 'mae', 'rmse', 'rel_mae', 'bias']].to_string(index=False))

    print("\n--- KEY FINDINGS ---")
    if len(df_ranges) > 0:
        hardest = df_ranges.loc[df_ranges['mae'].idxmax()]
        easiest = df_ranges.loc[df_ranges['mae'].idxmin()]
        print(f"Hardest count range: {hardest['count_range']} ({hardest['range_bounds']}) - MAE: {hardest['mae']:.2f}")
        print(f"Easiest count range: {easiest['count_range']} ({easiest['range_bounds']}) - MAE: {easiest['mae']:.2f}")

        # Check for systematic bias
        high_count = df_ranges[df_ranges['count_range'].isin(['high', 'very_high'])]
        if len(high_count) > 0 and high_count['bias'].mean() < -5:
            print(f"⚠️  Systematic UNDERCOUNTING in high-count images (avg bias: {high_count['bias'].mean():.1f})")
        elif len(high_count) > 0 and high_count['bias'].mean() > 5:
            print(f"⚠️  Systematic OVERCOUNTING in high-count images (avg bias: {high_count['bias'].mean():.1f})")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
