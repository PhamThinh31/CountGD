"""
Density-Guided Hybrid Counting for CountGD

Combines detection-based counting with density map estimation for
more robust object counting. This is the D.1 contribution.

Approaches:
1. Ensemble: Average detection count and density integral
2. Confidence-weighted: Weight by prediction confidence
3. Adaptive: Use density for high-count, detection for low-count
4. Count consistency: Regularize detection with density

This script evaluates these hybrid approaches.
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
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from main_inference import get_args_parser
from util.slconfig import SLConfig


def load_model_and_data(args):
    """Load model and dataset."""
    from datasets import build_dataset
    from models import build_model
    from util.misc import clean_state_dict

    # Enable density head
    args.use_density_head = True

    model, criterion, postprocessors = build_model(args)
    model.to(args.device)
    model.eval()

    checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']

    # Load with strict=False to allow missing density head weights
    model.load_state_dict(clean_state_dict(checkpoint), strict=False)

    dataset_val = build_dataset(image_set='val', args=args)

    return model, dataset_val, postprocessors


def run_inference_with_density(model, sample, exemplars, caption, args, device):
    """Run model inference and return detection + density outputs."""
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


def get_detection_count(outputs, args, tokenized_caption):
    """Get count from detection head."""
    logits = outputs['pred_logits'][0].sigmoid()

    end_idx = 1
    for token_ind in range(len(tokenized_caption)):
        if tokenized_caption[token_ind] == 1012:
            end_idx = token_ind
            break

    box_mask = logits.max(dim=-1).values > args.box_threshold
    filtered_logits = logits[box_mask]

    if filtered_logits.shape[0] > 0:
        text_mask = (filtered_logits[:, 1:end_idx] > args.text_threshold).sum(dim=-1) == (end_idx - 1)
        det_count = text_mask.sum().item()
        # Get average confidence of detected boxes
        det_confidence = filtered_logits[text_mask].max(dim=-1).values.mean().item() if det_count > 0 else 0
    else:
        det_count = 0
        det_confidence = 0

    return det_count, det_confidence


def get_density_count(outputs):
    """Get count from density map (integral of density)."""
    if 'density_pred' not in outputs:
        return 0, 0

    density_map = outputs['density_pred'][0]  # (1, H, W)
    density_count = density_map.sum().item()

    # Confidence: inverse of variance (more uniform = higher confidence)
    density_var = density_map.var().item()
    density_confidence = 1.0 / (1.0 + density_var)

    return density_count, density_confidence


def hybrid_ensemble(det_count, density_count, alpha=0.5):
    """Simple weighted average of detection and density counts."""
    return alpha * det_count + (1 - alpha) * density_count


def hybrid_confidence_weighted(det_count, det_conf, density_count, density_conf):
    """Weight by confidence scores."""
    total_conf = det_conf + density_conf
    if total_conf == 0:
        return det_count  # Fallback to detection

    w_det = det_conf / total_conf
    w_density = density_conf / total_conf

    return w_det * det_count + w_density * density_count


def hybrid_adaptive(det_count, density_count, threshold=50):
    """Use density for high-count images, detection for low-count."""
    # Use detection as initial estimate
    if det_count > threshold:
        # High count: trust density more (detection may miss objects)
        return 0.3 * det_count + 0.7 * density_count
    else:
        # Low count: trust detection more (density may be noisy)
        return 0.8 * det_count + 0.2 * density_count


def hybrid_max_confidence(det_count, det_conf, density_count, density_conf):
    """Use the count from the more confident source."""
    if det_conf > density_conf:
        return det_count
    else:
        return density_count


def get_category_name(dataset, target):
    """Extract category name from target."""
    if 'category_name' in target:
        return target['category_name']
    elif 'labels' in target and hasattr(dataset, 'cat_list'):
        label_idx = target['labels'][0].item()
        if label_idx < len(dataset.cat_list):
            return dataset.cat_list[label_idx]
    return 'unknown'


def evaluate_hybrid_methods(model, dataset, args, device):
    """Evaluate different hybrid counting methods."""
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_type)

    # Results storage
    results = {
        'detection_only': [],
        'density_only': [],
        'ensemble_0.3': [],
        'ensemble_0.5': [],
        'ensemble_0.7': [],
        'confidence_weighted': [],
        'adaptive': [],
        'max_confidence': [],
    }

    gt_counts = []
    categories = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating {len(dataset)} images with hybrid methods...")

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

        outputs = run_inference_with_density(model, sample, exemplars, caption, args, device)

        # Get counts from both sources
        det_count, det_conf = get_detection_count(outputs, args, tokenized['input_ids'][0])
        density_count, density_conf = get_density_count(outputs)

        # Get ground truth
        if 'labels_uncropped' in target:
            gt_count = target['labels_uncropped'].shape[0]
        else:
            gt_count = target['boxes'].shape[0]

        gt_counts.append(gt_count)
        categories.append(category)

        # Compute hybrid predictions
        results['detection_only'].append(det_count)
        results['density_only'].append(density_count)
        results['ensemble_0.3'].append(hybrid_ensemble(det_count, density_count, 0.3))
        results['ensemble_0.5'].append(hybrid_ensemble(det_count, density_count, 0.5))
        results['ensemble_0.7'].append(hybrid_ensemble(det_count, density_count, 0.7))
        results['confidence_weighted'].append(
            hybrid_confidence_weighted(det_count, det_conf, density_count, density_conf)
        )
        results['adaptive'].append(hybrid_adaptive(det_count, density_count))
        results['max_confidence'].append(
            hybrid_max_confidence(det_count, det_conf, density_count, density_conf)
        )

    return results, gt_counts, categories


def compute_metrics(predictions, gt_counts):
    """Compute MAE and RMSE."""
    predictions = np.array(predictions)
    gt_counts = np.array(gt_counts)
    errors = np.abs(predictions - gt_counts)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    return mae, rmse


def main():
    parser = argparse.ArgumentParser('Density Hybrid Evaluation', parents=[get_args_parser()])
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

    # Ensure density head is enabled
    args.use_density_head = True

    print("Loading model and data...")
    model, dataset, postprocessors = load_model_and_data(args)

    # Check if model has density head
    if not hasattr(model, 'density_head'):
        print("WARNING: Model does not have density head. Training a model with use_density_head=True required.")
        print("Running detection-only evaluation...")

    print("Evaluating hybrid methods...")
    results, gt_counts, categories = evaluate_hybrid_methods(model, dataset, args, device)

    # Compute metrics for each method
    metrics = []
    for method, predictions in results.items():
        mae, rmse = compute_metrics(predictions, gt_counts)
        metrics.append({
            'method': method,
            'mae': mae,
            'rmse': rmse,
        })

    df_metrics = pd.DataFrame(metrics).sort_values('mae')
    df_metrics.to_csv(os.path.join(args.output_dir, 'hybrid_methods_results.csv'), index=False)

    # Save detailed predictions
    df_detailed = pd.DataFrame({
        'gt_count': gt_counts,
        'category': categories,
        **{method: preds for method, preds in results.items()}
    })
    df_detailed.to_csv(os.path.join(args.output_dir, 'hybrid_predictions_detailed.csv'), index=False)

    # Print results
    print("\n" + "="*80)
    print("DENSITY-GUIDED HYBRID COUNTING RESULTS")
    print("="*80)

    print("\n--- METHOD COMPARISON ---")
    print(df_metrics.to_string(index=False))

    best_method = df_metrics.iloc[0]
    baseline_det = df_metrics[df_metrics['method'] == 'detection_only'].iloc[0]

    print(f"\n--- BEST METHOD ---")
    print(f"Method: {best_method['method']}")
    print(f"MAE: {best_method['mae']:.4f}")
    print(f"RMSE: {best_method['rmse']:.4f}")

    improvement = baseline_det['mae'] - best_method['mae']
    print(f"\n--- IMPROVEMENT OVER DETECTION-ONLY ---")
    print(f"Detection-only MAE: {baseline_det['mae']:.4f}")
    print(f"Best hybrid MAE: {best_method['mae']:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement/baseline_det['mae']*100:.2f}%)")

    # Per count-range analysis
    print("\n--- PER COUNT-RANGE ANALYSIS ---")
    df_detailed['count_range'] = pd.cut(df_detailed['gt_count'],
                                        bins=[0, 10, 50, 100, float('inf')],
                                        labels=['low', 'medium', 'high', 'very_high'])

    for count_range in ['low', 'medium', 'high', 'very_high']:
        subset = df_detailed[df_detailed['count_range'] == count_range]
        if len(subset) == 0:
            continue

        print(f"\n  {count_range.upper()} COUNT ({len(subset)} images):")
        for method in ['detection_only', 'density_only', best_method['method']]:
            if method in subset.columns:
                mae = np.abs(subset[method] - subset['gt_count']).mean()
                print(f"    {method}: MAE = {mae:.2f}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
