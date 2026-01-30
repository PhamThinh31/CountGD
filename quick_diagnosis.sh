#!/bin/bash

# Quick diagnosis of COCO dataset issues
# Usage: bash quick_diagnosis.sh path/to/val_coco.json

COCO_FILE=${1:-"./custom_dataset/od_v1.1/val_coco.json"}

echo "================================================================================"
echo "QUICK COCO DATASET DIAGNOSIS"
echo "================================================================================"
echo ""
echo "File: $COCO_FILE"
echo ""

# Check if file exists
if [ ! -f "$COCO_FILE" ]; then
    echo "❌ Error: File not found: $COCO_FILE"
    exit 1
fi

# Quick Python check
python3 - <<EOF
import json
from collections import defaultdict

with open('$COCO_FILE', 'r') as f:
    data = json.load(f)

images = data['images']
anns = data['annotations']
categories = {c['id']: c['name'] for c in data['categories']}

print(f"📊 Dataset Overview:")
print(f"   Total Images: {len(images)}")
print(f"   Total Annotations: {len(anns)}")
print(f"   Categories: {list(categories.values())}")
print()

# Group by image
img_anns = defaultdict(list)
for ann in anns:
    img_anns[ann['image_id']].append(ann)

# Check annotation counts
ann_counts = [len(img_anns[img['id']]) for img in images]
point_counts = []
exemplar_counts = []

for img in images:
    img_id = img['id']
    img_anns_list = img_anns[img_id]
    exemplars = [a for a in img_anns_list if a.get('area', 0) != 4]
    points = [a for a in img_anns_list if a.get('area', 0) == 4]
    point_counts.append(len(points))
    exemplar_counts.append(len(exemplars))

print(f"📈 Annotations per Image:")
print(f"   Min: {min(ann_counts)}")
print(f"   Max: {max(ann_counts)}")
print(f"   Mean: {sum(ann_counts)/len(ann_counts):.1f}")
print()

print(f"📍 Point Annotations (area=4) per Image:")
print(f"   Min: {min(point_counts)}")
print(f"   Max: {max(point_counts)}")
print(f"   Mean: {sum(point_counts)/len(point_counts):.1f}")
print()

print(f"🔷 Exemplar Annotations (area!=4) per Image:")
print(f"   Min: {min(exemplar_counts)}")
print(f"   Max: {max(exemplar_counts)}")
print(f"   Mean: {sum(exemplar_counts)/len(exemplar_counts):.1f}")
print()

# Find problematic images
problems = []
for img in images:
    img_id = img['id']
    img_anns_list = img_anns[img_id]
    exemplars = [a for a in img_anns_list if a.get('area', 0) != 4]
    points = [a for a in img_anns_list if a.get('area', 0) == 4]

    if len(img_anns_list) > 500:
        problems.append({
            'file': img['file_name'],
            'total': len(img_anns_list),
            'points': len(points),
            'exemplars': len(exemplars),
            'issue': 'too_many_annotations'
        })
    elif len(exemplars) != 3:
        problems.append({
            'file': img['file_name'],
            'total': len(img_anns_list),
            'points': len(points),
            'exemplars': len(exemplars),
            'issue': 'wrong_exemplar_count'
        })

if problems:
    print(f"⚠️  Found {len(problems)} images with issues:")
    print()
    for p in problems[:10]:
        print(f"   {p['issue']:25s} | {p['file']:40s}")
        print(f"   {'':25s} | Total: {p['total']}, Points: {p['points']}, Exemplars: {p['exemplars']}")
        print()

    print(f"\n🔍 For detailed check, run:")
    print(f"   python quick_check_and_fix.py --coco_file $COCO_FILE --check_all")
else:
    print("✅ No obvious issues found!")

EOF

echo ""
echo "================================================================================"
