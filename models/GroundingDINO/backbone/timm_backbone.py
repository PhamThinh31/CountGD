"""
Generic timm backbone wrapper for hierarchical models.
Supports Swin-V2-B, ConvNeXt-V2-B, FocalNet-B, and other multi-scale architectures
available in the timm library.

All these backbones produce multi-scale features NATIVELY (no FPN needed),
have ~88M params (same as Swin-B), and fit on a 3090 (24GB VRAM).

Usage in config:
    backbone = "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"   # Swin-V2-B
    backbone = "convnextv2_base.fcmae_ft_in22k_in1k_384"              # ConvNeXt-V2-B
    backbone = "focalnet_base_lrf.in1k"                                # FocalNet-B
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from groundingdino.util.misc import NestedTensor


class TimmBackbone(nn.Module):
    """
    Generic wrapper for timm hierarchical backbones that produce multi-scale features.
    Matches the interface expected by CountGD's Joiner/backbone system:
      - forward(NestedTensor) -> Dict[str, NestedTensor]
      - .num_channels: List[int] of output channel counts per level
      - .num_features: same (with padding for Swin indexing compatibility)
    """

    def __init__(
        self,
        model_name,
        pretrained=True,
        out_indices=(1, 2, 3),
        freeze=True,
    ):
        super().__init__()
        # Create timm feature extractor with multi-scale output
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,        # Return intermediate features
            out_indices=list(out_indices),
        )

        # Get the output channel counts from timm
        # timm's features_only mode provides .feature_info
        feature_info = self.body.feature_info.channels()
        self.num_channels = list(feature_info)
        # Pad for Swin-style indexing: num_features[4 - len(indices):]
        padding = [0] * (4 - len(self.num_channels))
        self.num_features = padding + self.num_channels

        if freeze:
            for param in self.body.parameters():
                param.requires_grad = False

        print(f"[TimmBackbone] Loaded {model_name}")
        print(f"  out_indices={list(out_indices)}, channels={self.num_channels}")
        n_params = sum(p.numel() for p in self.body.parameters()) / 1e6
        n_trainable = sum(p.numel() for p in self.body.parameters() if p.requires_grad) / 1e6
        print(f"  params={n_params:.1f}M, trainable={n_trainable:.1f}M")

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor with tensors (B, 3, H, W) and mask (B, H, W)
        Returns:
            Dict[str, NestedTensor] — one entry per feature level
        """
        x = tensor_list.tensors
        mask = tensor_list.mask

        # timm features_only forward returns a list of feature maps
        features = self.body(x)

        out = {}
        for i, feat in enumerate(features):
            # Create downsampled mask matching this feature level's spatial size
            feat_mask = F.interpolate(
                mask[None].float(), size=feat.shape[-2:]
            ).to(torch.bool)[0]
            out[str(i)] = NestedTensor(feat, feat_mask)

        return out
