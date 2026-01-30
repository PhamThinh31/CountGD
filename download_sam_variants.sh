#!/bin/bash

mkdir -p checkpoints

echo "Downloading SAM ViT-H (already should have this - 2.4GB)..."
if [ ! -f "checkpoints/sam_vit_h_4b8939.pth" ]; then
    wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo "SAM ViT-H already exists, skipping..."
fi

echo ""
echo "Downloading SAM ViT-L (1.2GB)..."
if [ ! -f "checkpoints/sam_vit_l_0b3195.pth" ]; then
    wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
else
    echo "SAM ViT-L already exists, skipping..."
fi

echo ""
echo "Downloading SAM ViT-B (375MB)..."
if [ ! -f "checkpoints/sam_vit_b_01ec64.pth" ]; then
    wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
else
    echo "SAM ViT-B already exists, skipping..."
fi

echo ""
echo "All SAM variants downloaded!"
echo ""
echo "Model sizes:"
du -h checkpoints/sam_vit_*.pth
