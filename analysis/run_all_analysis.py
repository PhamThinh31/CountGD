#!/usr/bin/env python
"""
Run All Analysis Scripts for CountGD

This script orchestrates running all analysis components and generates
a comprehensive report for the thesis.

Usage:
    python analysis/run_all_analysis.py \
        -c config/cfg_fsc147_vit_b.py \
        --datasets config/datasets_fsc147.json \
        --pretrain_model_path /path/to/checkpoint.pth \
        --output_dir ./analysis_results \
        --options text_encoder_type=checkpoints/bert-base-uncased
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def run_script(script_name, args, output_dir):
    """Run an analysis script and return success status."""
    script_path = Path(__file__).parent / script_name

    cmd = [
        sys.executable, str(script_path),
        '-c', args.config_file,
        '--datasets', args.datasets,
        '--pretrain_model_path', args.pretrain_model_path,
        '--output_dir', output_dir,
    ]

    if args.options:
        cmd.extend(['--options'] + args.options)

    if hasattr(args, 'device') and args.device:
        cmd.extend(['--device', args.device])

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False


def generate_summary_report(output_dir, args):
    """Generate a comprehensive summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COUNTGD COMPREHENSIVE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Checkpoint: {args.pretrain_model_path}")
    report_lines.append(f"Config: {args.config_file}")

    # Load and summarize results from each analysis

    # 1. Per-category results
    per_cat_path = os.path.join(output_dir, 'per_category_results.csv')
    if os.path.exists(per_cat_path):
        df = pd.read_csv(per_cat_path)
        report_lines.append("\n" + "-" * 40)
        report_lines.append("1. PER-CATEGORY ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total categories evaluated: {len(df)}")
        report_lines.append(f"Overall MAE: {df['mae'].mean():.2f}")

        report_lines.append("\nTop 5 HARDEST categories:")
        for _, row in df.nlargest(5, 'mae').iterrows():
            report_lines.append(f"  - {row['category']}: MAE={row['mae']:.2f}, n={row['n_samples']}")

        report_lines.append("\nTop 5 EASIEST categories:")
        for _, row in df.nsmallest(5, 'mae').iterrows():
            report_lines.append(f"  - {row['category']}: MAE={row['mae']:.2f}, n={row['n_samples']}")

    # 2. Count-stratified results
    count_strat_path = os.path.join(output_dir, 'count_stratified_results.csv')
    if os.path.exists(count_strat_path):
        df = pd.read_csv(count_strat_path)
        report_lines.append("\n" + "-" * 40)
        report_lines.append("2. COUNT-STRATIFIED ANALYSIS")
        report_lines.append("-" * 40)

        for _, row in df.iterrows():
            report_lines.append(f"\n{row['count_range'].upper()} ({row['range_bounds']}):")
            report_lines.append(f"  Samples: {row['n_samples']}")
            report_lines.append(f"  MAE: {row['mae']:.2f}")
            report_lines.append(f"  RMSE: {row['rmse']:.2f}")
            report_lines.append(f"  Relative MAE: {row['rel_mae']:.2%}")
            report_lines.append(f"  Bias: {row['bias']:+.2f}")

    # 3. Failure analysis
    failure_path = os.path.join(output_dir, 'failure_patterns.json')
    if os.path.exists(failure_path):
        with open(failure_path, 'r') as f:
            patterns = json.load(f)

        report_lines.append("\n" + "-" * 40)
        report_lines.append("3. FAILURE ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total images: {patterns['total_images']}")
        report_lines.append(f"Mean error: {patterns['mean_error']:.2f}")
        report_lines.append(f"Median error: {patterns['median_error']:.2f}")

        if 'error_distribution' in patterns:
            report_lines.append("\nError type distribution:")
            for error_type, count in patterns['error_distribution'].items():
                pct = count / patterns['total_images'] * 100
                report_lines.append(f"  - {error_type}: {count} ({pct:.1f}%)")

    # 4. Hybrid methods results
    hybrid_path = os.path.join(output_dir, 'hybrid_methods_results.csv')
    if os.path.exists(hybrid_path):
        df = pd.read_csv(hybrid_path)
        report_lines.append("\n" + "-" * 40)
        report_lines.append("4. DENSITY-GUIDED HYBRID METHODS")
        report_lines.append("-" * 40)

        for _, row in df.iterrows():
            report_lines.append(f"  {row['method']}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}")

        best = df.iloc[0]
        det_only = df[df['method'] == 'detection_only'].iloc[0]
        improvement = det_only['mae'] - best['mae']

        report_lines.append(f"\nBest method: {best['method']}")
        report_lines.append(f"Improvement over detection: {improvement:.4f} ({improvement/det_only['mae']*100:.2f}%)")

    # Key findings and recommendations
    report_lines.append("\n" + "=" * 80)
    report_lines.append("KEY FINDINGS & RECOMMENDATIONS")
    report_lines.append("=" * 80)

    report_lines.append("""
Based on the analysis:

1. CATEGORY DIFFICULTY:
   - Some categories are systematically harder than others
   - This may be due to: object size, visual similarity, typical count range

2. COUNT RANGE PERFORMANCE:
   - Model typically performs better on low-count images
   - High-count images show systematic undercounting (limited by num_queries)

3. FAILURE PATTERNS:
   - Overcounting: Often in images with repetitive background patterns
   - Undercounting: Often in dense scenes with occluded objects

4. HYBRID METHODS:
   - Combining detection and density can improve robustness
   - Best approach depends on the count range of the image

THESIS CONTRIBUTIONS:
- Comprehensive analysis of CountGD's strengths and weaknesses
- Identification of failure modes by category and count range
- Novel density-guided hybrid counting approach
- Recommendations for improving detection-based counting
""")

    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser('Run All CountGD Analysis')
    parser.add_argument('-c', '--config_file', required=True, help='Config file path')
    parser.add_argument('--datasets', required=True, help='Datasets config path')
    parser.add_argument('--pretrain_model_path', required=True, help='Checkpoint path')
    parser.add_argument('--output_dir', default='./analysis_results', help='Output directory')
    parser.add_argument('--options', nargs='+', help='Additional options')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument('--skip_density', action='store_true', help='Skip density hybrid analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("COUNTGD COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint: {args.pretrain_model_path}")

    # Run each analysis script
    scripts = [
        ('per_category_eval.py', 'Per-category evaluation'),
        ('count_stratified_eval.py', 'Count-stratified evaluation'),
        ('failure_analysis.py', 'Failure analysis'),
    ]

    if not args.skip_density:
        scripts.append(('density_hybrid.py', 'Density hybrid methods'))

    results = {}
    for script, description in scripts:
        print(f"\n>>> {description}...")
        success = run_script(script, args, args.output_dir)
        results[script] = success
        if success:
            print(f"✓ {description} completed")
        else:
            print(f"✗ {description} failed")

    # Generate summary report
    print("\n>>> Generating summary report...")
    generate_summary_report(args.output_dir, args)

    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nGenerated files:")
    for f in os.listdir(args.output_dir):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
