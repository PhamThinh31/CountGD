"""
EVA-02 ViT-L backbone wrapper with FPN for multi-scale feature generation.
Produces 3 feature levels compatible with CountGD's deformable transformer.

EVA-02 ViT-L outputs single-scale features (1024-dim at ~1/14 resolution).
This module adds a lightweight FPN to produce multi-scale features matching
Swin-B's output interface: channels [256, 512, 1024] at scales [1/8, 1/16, 1/32].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from groundingdino.util.misc import NestedTensor


class EVA02FPN(nn.Module):
    """
    Feature Pyramid Network that creates multi-scale features from
    EVA-02's single-scale ViT output.

    Produces 3 scales to match Swin-B's [256, 512, 1024] channel interface:
      - Level 0 (1/8 scale):  256 channels  (upsampled from ViT output)
      - Level 1 (1/16 scale): 512 channels  (derived from ViT output)
      - Level 2 (1/32 scale): 1024 channels (downsampled from ViT output)
    """

    def __init__(self, vit_dim=1024, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = [256, 512, 1024]
        self.out_channels = out_channels

        # Level 0: Upsample 2x for ~1/8 scale
        self.lateral_conv_0 = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, out_channels[0], kernel_size=2, stride=2),
            nn.GroupNorm(32, out_channels[0]),
            nn.GELU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels[0]),
        )

        # Level 1: Keep at ~1/14-1/16 scale
        self.lateral_conv_1 = nn.Sequential(
            nn.Conv2d(vit_dim, out_channels[1], kernel_size=1),
            nn.GroupNorm(32, out_channels[1]),
        )

        # Level 2: Downsample 2x for ~1/32 scale
        self.lateral_conv_2 = nn.Sequential(
            nn.Conv2d(vit_dim, out_channels[2], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels[2]),
        )

    def forward(self, vit_features, H, W):
        """
        Args:
            vit_features: (B, N, C) where N = H*W tokens, C = vit_dim
            H, W: spatial dimensions of the token grid
        Returns:
            list of 3 feature maps at different scales
        """
        B, N, C = vit_features.shape
        x = vit_features.transpose(1, 2).reshape(B, C, H, W)

        feat_0 = self.lateral_conv_0(x)  # ~1/8 scale, 256ch
        feat_1 = self.lateral_conv_1(x)  # ~1/16 scale, 512ch
        feat_2 = self.lateral_conv_2(x)  # ~1/32 scale, 1024ch

        return [feat_0, feat_1, feat_2]


class EVA02Backbone(nn.Module):
    """
    EVA-02 ViT-L backbone that outputs multi-scale NestedTensors,
    matching the interface of the existing Swin backbone.

    The ViT body is optionally frozen (to preserve pretrained features),
    while the FPN layers are always trainable.

    Compatible with existing CountGD code:
      - Returns Dict[str, NestedTensor] from forward()
      - Exposes .num_channels and .num_features attributes
      - Works with Joiner, build_position_encoding, input_proj layers
    """

    def __init__(
        self,
        model_name='eva02_large_patch14_448',
        pretrained=True,
        out_channels=None,
        freeze=True,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = [256, 512, 1024]

        # Load EVA-02 via timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # Remove classification head
            global_pool='',      # No global pooling -- keep spatial tokens
        )
        self.vit_dim = self.vit.embed_dim  # 1024 for EVA-02-L
        self.patch_size = self._get_patch_size()

        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        # FPN to produce multi-scale features
        self.fpn = EVA02FPN(vit_dim=self.vit_dim, out_channels=out_channels)

        # Interface attributes matching Swin backbone
        self.num_channels = out_channels  # [256, 512, 1024]
        # Padded to match Swin indexing: num_features[4 - len(return_interm_indices):]
        # With return_interm_indices=[1,2,3], backbone.py does num_features[1:]
        self.num_features = [0] + out_channels  # [0, 256, 512, 1024]

    def _get_patch_size(self):
        """Extract patch size from the ViT's patch embedding."""
        if hasattr(self.vit, 'patch_embed'):
            pe = self.vit.patch_embed
            if hasattr(pe, 'patch_size'):
                ps = pe.patch_size
                return ps[0] if isinstance(ps, (tuple, list)) else ps
            if hasattr(pe, 'proj') and hasattr(pe.proj, 'kernel_size'):
                return pe.proj.kernel_size[0]
        return 14  # Default for EVA-02

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor with:
                - tensors: [B, 3, H, W]
                - mask: [B, H, W] (True = padded)
        Returns:
            out: Dict[str, NestedTensor] with 3 feature levels
        """
        x = tensor_list.tensors  # (B, 3, H_img, W_img)
        mask = tensor_list.mask   # (B, H_img, W_img)
        B, _, H_img, W_img = x.shape

        # EVA-02 forward through patch embedding
        x = self.vit.patch_embed(x)  # (B, N, C)

        # Handle position embedding
        if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
            x = x + self._interpolate_pos_embed(x, H_img, W_img)

        if hasattr(self.vit, 'pos_drop'):
            x = self.vit.pos_drop(x)

        # Handle cls_token if present
        has_cls_token = hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None
        if has_cls_token:
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # Forward through all ViT blocks
        for blk in self.vit.blocks:
            x = blk(x)

        if hasattr(self.vit, 'norm'):
            x = self.vit.norm(x)

        # Remove cls token if present
        if has_cls_token:
            x = x[:, 1:, :]

        # Compute spatial grid size
        H = H_img // self.patch_size
        W = W_img // self.patch_size

        # Ensure token count matches grid
        if x.shape[1] != H * W:
            # Handle fractional patches -- take only the valid tokens
            x = x[:, :H * W, :]

        # Generate multi-scale features via FPN
        multi_scale_feats = self.fpn(x, H, W)

        # Wrap in NestedTensor format with appropriate masks
        out = {}
        for i, feat in enumerate(multi_scale_feats):
            feat_mask = F.interpolate(
                mask[None].float(), size=feat.shape[-2:]
            ).to(torch.bool)[0]
            out[str(i)] = NestedTensor(feat, feat_mask)

        return out

    def _interpolate_pos_embed(self, x, H_img, W_img):
        """Interpolate position embeddings to match actual patch count."""
        pos_embed = self.vit.pos_embed
        N = x.shape[1]  # Number of patch tokens

        # Check if pos_embed has cls_token slot
        has_cls = pos_embed.shape[1] > N

        if has_cls:
            patch_pos = pos_embed[:, 1:]
        else:
            patch_pos = pos_embed

        if patch_pos.shape[1] == N:
            return patch_pos if not has_cls else patch_pos

        # Need interpolation for variable-size inputs
        dim = patch_pos.shape[-1]
        orig_size = int(patch_pos.shape[1] ** 0.5)
        new_H = H_img // self.patch_size
        new_W = W_img // self.patch_size

        patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(new_H, new_W), mode='bicubic', align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return patch_pos
