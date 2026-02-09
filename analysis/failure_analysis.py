"""
Failure Case Analysis for CountGD

Identifies and analyzes the worst failure cases to understand
when and why the model fails.

Outputs:
- Top K failure cases with metadata
- Failure patterns (by category, count range, etc.)
- Visualization of failure cases
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from main_inference import get_args_parser
from util.slconfig import SLConfig


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


def run_inference_with_boxes(model, sample, exemplars, caption, args, device):
    """Run model inference and return boxes and scores."""
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


def get_predictions(outputs, args, tokenized_caption, img_size):
    """Get predicted boxes and count."""
    logits = outputs['pred_logits'][0].sigmoid()
    boxes = outputs['pred_boxes'][0]

    end_idx = 1
    for token_ind in range(len(tokenized_caption)):
        if tokenized_caption[token_ind] == 1012:
            end_idx = token_ind
            break

    # Apply thresholds
    box_mask = logits.max(dim=-1).values > args.box_threshold
    filtered_logits = logits[box_mask]
    filtered_boxes = boxes[box_mask]

    if filtered_logits.shape[0] > 0:
        text_mask = (filtered_logits[:, 1:end_idx] > args.text_threshold).sum(dim=-1) == (end_idx - 1)
        final_boxes = filtered_boxes[text_mask]
        final_scores = filtered_logits[text_mask].max(dim=-1).values
    else:
        final_boxes = torch.zeros(0, 4)
        final_scores = torch.zeros(0)

    # Convert to pixel coordinates
    h, w = img_size
    if final_boxes.shape[0] > 0:
        # cxcywh to xyxy
        cx, cy, bw, bh = final_boxes.unbind(-1)
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
        final_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    return final_boxes.cpu(), final_scores.cpu(), final_boxes.shape[0]


def get_category_name(dataset, target):
    """Extract category name from target."""
    if 'category_name' in target:
        return target['category_name']
    elif 'labels' in target and hasattr(dataset, 'cat_list'):
        label_idx = target['labels'][0].item()
        if label_idx < len(dataset.cat_list):
            return dataset.cat_list[label_idx]
    return 'unknown'


def analyze_failures(model, dataset, args, device, top_k=50):
    """Analyze failure cases."""
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_type)

    all_results = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Analyzing {len(dataset)} images...")

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

        # Get image size
        img_size = (sample.shape[1], sample.shape[2])  # H, W

        outputs = run_inference_with_boxes(model, sample, exemplars, caption, args, device)
        pred_boxes, pred_scores, pred_count = get_predictions(
            outputs, args, tokenized['input_ids'][0], img_size
        )

        if 'labels_uncropped' in target:
            gt_count = target['labels_uncropped'].shape[0]
        else:
            gt_count = target['boxes'].shape[0]

        error = abs(gt_count - pred_count)
        signed_error = pred_count - gt_count

        # Get image path if available
        image_path = None
        if hasattr(dataset, 'coco') and hasattr(dataset.coco, 'loadImgs'):
            img_info = dataset.coco.loadImgs(target.get('image_id', idx).item())[0]
            image_path = os.path.join(dataset.img_folder, img_info['file_name'])

        all_results.append({
            'image_idx': idx,
            'image_id': target.get('image_id', idx).item() if torch.is_tensor(target.get('image_id', idx)) else idx,
            'image_path': image_path,
            'category': category,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'error': error,
            'signed_error': signed_error,
            'error_type': 'overcount' if signed_error > 0 else 'undercount' if signed_error < 0 else 'correct',
            'relative_error': error / gt_count if gt_count > 0 else 0,
            'pred_boxes': pred_boxes.numpy().tolist(),
            'pred_scores': pred_scores.numpy().tolist(),
            'exemplar_count': exemplars.shape[0],
        })

    # Sort by error
    all_results.sort(key=lambda x: x['error'], reverse=True)

    return all_results[:top_k], all_results


def analyze_failure_patterns(all_results):
    """Analyze patterns in failure cases."""
    df = pd.DataFrame(all_results)

    patterns = {
        'total_images': len(df),
        'mean_error': df['error'].mean(),
        'median_error': df['error'].median(),
    }

    # Error type distribution
    error_types = df['error_type'].value_counts().to_dict()
    patterns['error_distribution'] = error_types

    # Overcounting vs undercounting
    overcounts = df[df['signed_error'] > 0]
    undercounts = df[df['signed_error'] < 0]

    patterns['overcount_stats'] = {
        'count': len(overcounts),
        'mean_error': overcounts['error'].mean() if len(overcounts) > 0 else 0,
        'worst_categories': overcounts.groupby('category')['error'].mean().nlargest(5).to_dict()
    }

    patterns['undercount_stats'] = {
        'count': len(undercounts),
        'mean_error': undercounts['error'].mean() if len(undercounts) > 0 else 0,
        'worst_categories': undercounts.groupby('category')['error'].mean().nlargest(5).to_dict()
    }

    # Count range analysis
    df['count_range'] = pd.cut(df['gt_count'], bins=[0, 10, 50, 100, float('inf')],
                               labels=['low (1-10)', 'medium (11-50)', 'high (51-100)', 'very_high (100+)'])
    count_range_errors = df.groupby('count_range')['error'].agg(['mean', 'std', 'count']).to_dict()
    patterns['count_range_analysis'] = count_range_errors

    # Category analysis
    category_errors = df.groupby('category')['error'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    patterns['hardest_categories'] = category_errors.head(10).to_dict()
    patterns['easiest_categories'] = category_errors.tail(10).to_dict()

    return patterns


def visualize_failure_case(result, output_path, dataset=None):
    """Visualize a single failure case."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Try to load image
    if result['image_path'] and os.path.exists(result['image_path']):
        img = Image.open(result['image_path'])
        ax.imshow(img)

        # Draw predicted boxes
        for box, score in zip(result['pred_boxes'], result['pred_scores']):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
    else:
        ax.text(0.5, 0.5, 'Image not available', ha='center', va='center', fontsize=20)

    # Add title with info
    title = (f"Category: {result['category']}\n"
             f"GT Count: {result['gt_count']} | Pred Count: {result['pred_count']} | "
             f"Error: {result['error']} ({result['error_type']})")
    ax.set_title(title, fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser('Failure Analysis', parents=[get_args_parser()])
    parser.add_argument('--output_dir', default='./analysis_results', help='Output directory')
    parser.add_argument('--top_k', type=int, default=50, help='Number of top failure cases to analyze')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'failure_visualizations'), exist_ok=True)

    cfg = SLConfig.fromfile(args.config_file)
    for key, value in vars(args).items():
        if value is not None:
            setattr(cfg, key, value)
    args = cfg
    args.device = device

    print("Loading model and data...")
    model, dataset, postprocessors = load_model_and_data(args)

    print(f"Analyzing failures (top {args.top_k})...")
    top_failures, all_results = analyze_failures(model, dataset, args, device, args.top_k)

    # Analyze patterns
    print("Analyzing failure patterns...")
    patterns = analyze_failure_patterns(all_results)

    # Save results
    with open(os.path.join(args.output_dir, 'failure_cases.json'), 'w') as f:
        json.dump(top_failures, f, indent=2)

    with open(os.path.join(args.output_dir, 'failure_patterns.json'), 'w') as f:
        json.dump(patterns, f, indent=2, default=str)

    # Save all results as CSV
    df_all = pd.DataFrame(all_results)
    df_all = df_all.drop(columns=['pred_boxes', 'pred_scores'])  # Remove large columns
    df_all.to_csv(os.path.join(args.output_dir, 'all_predictions.csv'), index=False)

    # Visualize top failures
    if args.visualize:
        print("Generating visualizations...")
        for i, result in enumerate(tqdm(top_failures[:20])):  # Top 20
            viz_path = os.path.join(args.output_dir, 'failure_visualizations', f'failure_{i+1}.png')
            visualize_failure_case(result, viz_path, dataset)

    # Print summary
    print("\n" + "="*80)
    print("FAILURE ANALYSIS RESULTS")
    print("="*80)

    print(f"\n--- OVERALL STATISTICS ---")
    print(f"Total images: {patterns['total_images']}")
    print(f"Mean error: {patterns['mean_error']:.2f}")
    print(f"Median error: {patterns['median_error']:.2f}")

    print(f"\n--- ERROR TYPE DISTRIBUTION ---")
    for error_type, count in patterns['error_distribution'].items():
        pct = count / patterns['total_images'] * 100
        print(f"  {error_type}: {count} ({pct:.1f}%)")

    print(f"\n--- TOP 10 FAILURE CASES ---")
    for i, case in enumerate(top_failures[:10]):
        print(f"  {i+1}. [{case['category']}] GT: {case['gt_count']}, Pred: {case['pred_count']}, "
              f"Error: {case['error']} ({case['error_type']})")

    print(f"\n--- OVERCOUNTING ANALYSIS ---")
    print(f"  Total overcounts: {patterns['overcount_stats']['count']}")
    print(f"  Mean overcount error: {patterns['overcount_stats']['mean_error']:.2f}")
    print(f"  Worst categories for overcounting:")
    for cat, err in list(patterns['overcount_stats']['worst_categories'].items())[:5]:
        print(f"    - {cat}: {err:.2f}")

    print(f"\n--- UNDERCOUNTING ANALYSIS ---")
    print(f"  Total undercounts: {patterns['undercount_stats']['count']}")
    print(f"  Mean undercount error: {patterns['undercount_stats']['mean_error']:.2f}")
    print(f"  Worst categories for undercounting:")
    for cat, err in list(patterns['undercount_stats']['worst_categories'].items())[:5]:
        print(f"    - {cat}: {err:.2f}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
