"""
Quick diagnostic script to check dataset quality and find issues.

Usage:
    python check_dataset_quality.py --coco_file ./custom_dataset/od_v1.1/val_coco.json
"""

import argparse
import json
from collections import defaultdict
import numpy as np


def check_coco_dataset(coco_file):
    """Check COCO dataset for common issues."""

    print("=" * 80)
    print("COCO Dataset Quality Check")
    print("=" * 80)

    with open(coco_file, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    annotations = data['annotations']

    print(f"\n📊 Dataset Overview:")
    print(f"   Total Images: {len(images)}")
    print(f"   Total Annotations: {len(annotations)}")
    print(f"   Categories: {len(categories)}")

    # Group annotations by image
    image_anns = defaultdict(list)
    for ann in annotations:
        image_anns[ann['image_id']].append(ann)

    # Check for issues
    issues = {
        'no_annotations': [],
        'too_many_anns': [],
        'wrong_exemplar_count': [],
        'all_area_4': [],
        'no_area_4': [],
        'suspicious_boxes': []
    }

    print(f"\n🔍 Checking each image...")

    for img_id, img_info in images.items():
        anns = image_anns[img_id]

        # Issue 1: No annotations
        if len(anns) == 0:
            issues['no_annotations'].append(img_info['file_name'])
            continue

        # Count exemplars (area != 4) and points (area == 4)
        exemplars = [a for a in anns if a.get('area', 0) != 4]
        points = [a for a in anns if a.get('area', 0) == 4]

        # Issue 2: Wrong number of exemplars
        if len(exemplars) != 3:
            issues['wrong_exemplar_count'].append({
                'file_name': img_info['file_name'],
                'exemplar_count': len(exemplars),
                'point_count': len(points)
            })

        # Issue 3: All annotations are area=4 (no exemplars)
        if len(exemplars) == 0 and len(points) > 0:
            issues['all_area_4'].append({
                'file_name': img_info['file_name'],
                'point_count': len(points)
            })

        # Issue 4: No point annotations (area=4)
        if len(points) == 0 and len(exemplars) > 0:
            issues['no_area_4'].append({
                'file_name': img_info['file_name'],
                'exemplar_count': len(exemplars)
            })

        # Issue 5: Too many annotations (likely error)
        if len(anns) > 1000:
            issues['too_many_anns'].append({
                'file_name': img_info['file_name'],
                'total_anns': len(anns),
                'exemplars': len(exemplars),
                'points': len(points)
            })

        # Issue 6: Suspicious bounding boxes (too large or too small)
        for ann in anns:
            bbox = ann.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 4:
                w, h = bbox[2], bbox[3]
                img_w, img_h = img_info['width'], img_info['height']

                # Box is larger than 80% of image
                if w > img_w * 0.8 or h > img_h * 0.8:
                    issues['suspicious_boxes'].append({
                        'file_name': img_info['file_name'],
                        'reason': 'box too large',
                        'bbox': bbox,
                        'image_size': [img_w, img_h]
                    })
                # Box is smaller than 2x2 pixels
                elif w < 2 or h < 2:
                    # This is expected for area=4 point annotations
                    if ann.get('area', 0) != 4:
                        issues['suspicious_boxes'].append({
                            'file_name': img_info['file_name'],
                            'reason': 'box too small (not area=4)',
                            'bbox': bbox,
                            'area': ann.get('area', 0)
                        })

    # Print issues
    print(f"\n❌ Issues Found:")
    print(f"   Images with no annotations: {len(issues['no_annotations'])}")
    print(f"   Images with wrong exemplar count: {len(issues['wrong_exemplar_count'])}")
    print(f"   Images with all area=4 (no exemplars): {len(issues['all_area_4'])}")
    print(f"   Images with no area=4 (no points): {len(issues['no_area_4'])}")
    print(f"   Images with too many annotations: {len(issues['too_many_anns'])}")
    print(f"   Images with suspicious boxes: {len(issues['suspicious_boxes'])}")

    # Show details for critical issues
    if issues['wrong_exemplar_count']:
        print(f"\n⚠️  CRITICAL: Wrong Exemplar Count (should be exactly 3)")
        print(f"   Showing first 10:")
        for item in issues['wrong_exemplar_count'][:10]:
            print(f"   - {item['file_name']}: {item['exemplar_count']} exemplars, {item['point_count']} points")

    if issues['all_area_4']:
        print(f"\n⚠️  CRITICAL: All Annotations are area=4 (missing exemplars)")
        print(f"   Showing first 10:")
        for item in issues['all_area_4'][:10]:
            print(f"   - {item['file_name']}: {item['point_count']} points, 0 exemplars")

    if issues['too_many_anns']:
        print(f"\n⚠️  CRITICAL: Too Many Annotations (likely duplication error)")
        print(f"   Showing all:")
        for item in issues['too_many_anns']:
            print(f"   - {item['file_name']}: {item['total_anns']} total "
                  f"({item['exemplars']} exemplars + {item['points']} points)")

    # Statistics
    print(f"\n📈 Annotation Statistics:")

    point_counts = []
    exemplar_counts = []

    for img_id in images:
        anns = image_anns[img_id]
        exemplars = [a for a in anns if a.get('area', 0) != 4]
        points = [a for a in anns if a.get('area', 0) == 4]
        point_counts.append(len(points))
        exemplar_counts.append(len(exemplars))

    print(f"\n   Point Annotations (area=4) per image:")
    print(f"      Min: {np.min(point_counts)}")
    print(f"      Max: {np.max(point_counts)}")
    print(f"      Mean: {np.mean(point_counts):.1f}")
    print(f"      Median: {np.median(point_counts):.1f}")

    print(f"\n   Exemplar Annotations (area!=4) per image:")
    print(f"      Min: {np.min(exemplar_counts)}")
    print(f"      Max: {np.max(exemplar_counts)}")
    print(f"      Mean: {np.mean(exemplar_counts):.1f}")
    print(f"      Median: {np.median(exemplar_counts):.1f}")

    # Check category distribution
    print(f"\n📊 Category Distribution:")
    cat_counts = defaultdict(int)
    for ann in annotations:
        cat_id = ann['category_id']
        cat_counts[categories.get(cat_id, 'unknown')] += 1

    for cat_name, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat_name:30s}: {count:6d} annotations")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    total_issues = sum(len(v) if isinstance(v, list) else 0 for v in issues.values())

    if total_issues == 0:
        print("✅ No major issues found!")
    else:
        print(f"⚠️  Found {total_issues} images with issues")
        print(f"\nMost likely problems:")

        if issues['too_many_anns']:
            print(f"   🔴 CRITICAL: {len(issues['too_many_anns'])} images have excessive annotations")
            print(f"      → Check if convert_to_coco.py is duplicating annotations")

        if issues['all_area_4']:
            print(f"   🔴 CRITICAL: {len(issues['all_area_4'])} images missing exemplars")
            print(f"      → Re-run convert_to_coco.py to add exemplar annotations")

        if issues['wrong_exemplar_count']:
            print(f"   🟠 WARNING: {len(issues['wrong_exemplar_count'])} images have wrong exemplar count")
            print(f"      → Should have exactly 3 exemplars per image")

    print()

    return issues


def main():
    parser = argparse.ArgumentParser(description="Quick dataset quality check")
    parser.add_argument('--coco_file', type=str, required=True,
                       help='Path to COCO annotation file')

    args = parser.parse_args()

    issues = check_coco_dataset(args.coco_file)

    # Write issues to file
    import os
    output_file = os.path.join(os.path.dirname(args.coco_file), 'dataset_issues.json')
    with open(output_file, 'w') as f:
        json.dump(issues, f, indent=2)

    print(f"📄 Detailed issues saved to: {output_file}")


if __name__ == "__main__":
    main()
