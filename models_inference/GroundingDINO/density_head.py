"""
Auxiliary density map regression head for CountGD.
Operates on the encoder memory output to predict a 2D density map.
Supervised with Gaussian-smoothed dot maps from GT object centers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityHead(nn.Module):
    """
    Takes flattened encoder memory and spatial shapes,
    reconstructs the highest-resolution feature map,
    and predicts a density map via a small CNN.
    """

    def __init__(self, hidden_dim=256, num_conv_layers=3):
        super().__init__()
        layers = []
        for i in range(num_conv_layers - 1):
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
            ])
        # Final layer: single-channel density output
        layers.append(nn.Conv2d(hidden_dim, 1, kernel_size=1))
        self.conv_layers = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)  # Density must be non-negative

    def forward(self, memory, spatial_shapes, level_start_index):
        """
        Args:
            memory: (B, sum(H_i * W_i), C) -- encoder output
            spatial_shapes: (num_levels, 2) -- [(H0,W0), (H1,W1), ...]
            level_start_index: (num_levels,) -- start indices per level
        Returns:
            density_map: (B, 1, H0, W0) at the highest resolution level
        """
        B, _, C = memory.shape
        # Extract the highest-resolution level (level 0)
        H0, W0 = spatial_shapes[0][0].item(), spatial_shapes[0][1].item()
        start = level_start_index[0].item()
        end = start + H0 * W0
        feat = memory[:, start:end, :].transpose(1, 2).reshape(B, C, H0, W0)

        density = self.conv_layers(feat)
        density = self.relu(density)  # Non-negative density
        return density


def generate_density_target(targets, spatial_shape, sigma=3.0, device='cuda'):
    """
    Generate Gaussian-smoothed density map targets from GT box centers.

    Args:
        targets: list of dicts with 'boxes' key (cx, cy, w, h in [0,1] normalized)
        spatial_shape: (H, W) of the target density map
        sigma: Gaussian kernel sigma
    Returns:
        density_targets: (B, 1, H, W)
    """
    B = len(targets)
    H, W = spatial_shape
    density_maps = []

    for t in targets:
        boxes = t['boxes']  # (N, 4) in cxcywh normalized [0,1]
        dmap = torch.zeros(1, H, W, device=device)

        if boxes.shape[0] > 0:
            cx = (boxes[:, 0] * W).long().clamp(0, W - 1)
            cy = (boxes[:, 1] * H).long().clamp(0, H - 1)
            for i in range(len(cx)):
                dmap[0, cy[i], cx[i]] += 1.0

        # Apply Gaussian smoothing
        if sigma > 0 and dmap.sum() > 0:
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            dmap = dmap.unsqueeze(0)  # (1, 1, H, W)
            dmap = _gaussian_blur(dmap, kernel_size, sigma)
            dmap = dmap.squeeze(0)  # (1, H, W)

        density_maps.append(dmap)

    return torch.stack(density_maps)  # (B, 1, H, W)


def _gaussian_blur(x, kernel_size, sigma):
    """Apply Gaussian blur using a 2D convolution kernel."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    return F.conv2d(x, kernel_2d, padding=padding)
