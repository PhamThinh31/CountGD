import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random


def visualize_sample(entry, images_dir, output_path):
    """Visualize a single CountGD sample with annotations"""
    img_path = Path(images_dir) / entry['filename']

    if not img_path.exists():
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            test_path = Path(images_dir) / (Path(entry['filename']).stem + ext)
            if test_path.exists():
                img_path = test_path
                break

    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    img = Image.open(img_path)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)

    for inst in entry['detection']['instances']:
        x1, y1, x2, y2 = inst['bbox']
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=1, edgecolor='red', facecolor='none', alpha=0.7
        )
        ax.add_patch(rect)

        ax.plot([(x1 + x2) / 2], [(y1 + y2) / 2], 'ro', markersize=3)

    for i, exemplar in enumerate(entry['exemplars']):
        x1, y1, x2, y2 = exemplar
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

        ax.text(x1, y1 - 5, f'Ex{i+1}', color='lime', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    num_objects = len(entry['detection']['instances'])
    category = entry['detection']['instances'][0]['category'] if entry['detection']['instances'] else 'unknown'

    ax.set_title(f"{entry['filename']}\nCategory: {category}, Count: {num_objects}, Exemplars: {len(entry['exemplars'])}",
                 fontsize=12, weight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser("Visualize CountGD dataset")
    parser.add_argument('--jsonl', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    entries = []
    with open(args.jsonl, 'r') as f:
        for line in f:
            entries.append(json.loads(line))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    random.seed(args.random_seed)
    samples = random.sample(entries, min(args.num_samples, len(entries)))

    for i, entry in enumerate(samples):
        output_path = output_dir / f"sample_{i+1}_{Path(entry['filename']).stem}.png"
        visualize_sample(entry, args.images_dir, output_path)

    print(f"\nVisualized {len(samples)} samples in {output_dir}")


if __name__ == '__main__':
    main()
