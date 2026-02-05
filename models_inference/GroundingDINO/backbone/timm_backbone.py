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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from groundingdino.util.misc import NestedTensor


def _compute_swin_attn_mask(H, W, window_size, shift_size, device):
    """
    Recompute the shifted-window attention mask for a given spatial resolution.
    This is the same logic as SwinTransformerV2Block.__init__ but for arbitrary H, W.
    H, W must be divisible by window_size.
    """
    img_mask = torch.zeros((1, H, W, 1), device=device)
    cnt = 0
    for h in (
            slice(0, -window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None)):
        for w in (
                slice(0, -window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None)):
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # window_partition inline
    B, Hm, Wm, C = img_mask.shape
    img_mask = img_mask.view(B, Hm // window_size[0], window_size[0],
                             Wm // window_size[1], window_size[1], C)
    mask_windows = img_mask.permute(0, 1, 3, 2, 4, 5).contiguous()
    mask_windows = mask_windows.view(-1, window_size[0], window_size[1], C)

    window_area = window_size[0] * window_size[1]
    mask_windows = mask_windows.view(-1, window_area)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def _lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


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
        self.model_name = model_name
        self.is_swin = "swin" in model_name

        # Create timm feature extractor with multi-scale output
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,        # Return intermediate features
            out_indices=list(out_indices),
        )

        # Disable strict input size checks in PatchEmbed layers.
        self._disable_strict_img_size(self.body)

        # For Swin-V2: read the actual init-time resolution from each stage's
        # blocks to compute the correct pixel divisor robustly.
        if self.is_swin:
            self._stage_info = self._read_swin_stage_info(self.body)
            self.pixel_divisor = self._compute_swin_pixel_divisor(self._stage_info)
        else:
            self.pixel_divisor = 32

        # Get the output channel counts from timm
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
        print(f"  pixel_divisor={self.pixel_divisor}")
        if self.is_swin:
            for si in self._stage_info:
                print(f"  stage {si['name']}: init_block_res={si['init_block_res']}, "
                      f"window_size={si['window_size']}, "
                      f"pixel_factor={si['pixel_factor']}")
        n_params = sum(p.numel() for p in self.body.parameters()) / 1e6
        n_trainable = sum(p.numel() for p in self.body.parameters() if p.requires_grad) / 1e6
        print(f"  params={n_params:.1f}M, trainable={n_trainable:.1f}M")

    @staticmethod
    def _disable_strict_img_size(model):
        """Patch all PatchEmbed modules to accept variable input sizes."""
        from timm.layers import PatchEmbed
        for module in model.modules():
            if isinstance(module, PatchEmbed):
                module.img_size = None
                module.strict_img_size = False

    @staticmethod
    def _read_swin_stage_info(model):
        """
        Read the ACTUAL init-time block resolution from the model to compute
        the pixel-to-block-grid factor. This is robust because it uses the
        values timm already computed, rather than us guessing the architecture.

        For swinv2_base_window12to24 with img_size=384:
          layers_0 blocks: input_resolution=(96, 96)  → pixel_factor = 384/96 = 4
          layers_1 blocks: input_resolution=(48, 48)  → pixel_factor = 384/48 = 8
          layers_2 blocks: input_resolution=(24, 24)  → pixel_factor = 384/24 = 16
          layers_3 blocks: input_resolution=(12, 12)  → pixel_factor = 384/12 = 32
        """
        # First, find the model's init img_size from the PatchEmbed
        from timm.layers import PatchEmbed
        init_img_size = 384  # fallback
        for m in model.modules():
            if isinstance(m, PatchEmbed):
                # img_size may have been cleared by _disable_strict_img_size,
                # but we can infer from patch_size and the first stage's block resolution
                break

        info = []
        for name, module in model.named_modules():
            if type(module).__name__ != 'SwinTransformerV2Stage':
                continue
            if not hasattr(module, 'blocks') or len(module.blocks) == 0:
                continue

            block = module.blocks[0]
            block_res = block.input_resolution  # (H_grid, W_grid) at init time
            ws = block.window_size              # (ws_h, ws_w)

            info.append({
                'name': name,
                'stage': module,
                'init_block_res': block_res,
                'window_size': ws,
            })

        # Now compute pixel_factor for each stage.
        # pixel_factor = init_img_size / block_res[0]
        # We get init_img_size from: stage_0's block_res * patch_size
        # stage_0 always has the highest resolution = init_img_size / patch_size
        if info:
            # stage_0 block_res = init_img_size / patch_size
            # So init_img_size = stage_0_block_res * patch_size
            # And pixel_factor_for_stage_i = init_img_size / stage_i_block_res
            #                              = (stage_0_block_res / stage_i_block_res) * patch_size
            # But we can just use the ratio directly:
            # pixel_factor = stage_0_block_res[0] * patch_size / stage_i_block_res[0] ... no
            # Simpler: pixel_factor = how many pixels per grid cell at that stage
            # For stage_0: block_res=(96,96) with img=384 → factor = 384/96 = 4
            # We need to know the original img_size. We can get it:
            # original_img_size_h = stage_0_block_res[0] * (original_img / original_stage0_res)
            # Simplest: just use the first stage to find patch_size equivalent
            stage0_res = info[0]['init_block_res'][0]
            # patch_size = img_size / stage0_res. For 384/96=4
            # But we don't know img_size... unless we get it from PatchEmbed
            # Let's get patch_size directly
            patch_size = 4
            for m in model.modules():
                if isinstance(m, PatchEmbed):
                    ps = m.patch_size
                    patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
                    break
            # init_img_size = stage0_res * patch_size
            init_img_h = stage0_res * patch_size

            for si in info:
                si['pixel_factor'] = init_img_h // si['init_block_res'][0]

        return info

    @staticmethod
    def _compute_swin_pixel_divisor(stage_info):
        """
        Compute pixel_divisor = LCM of (pixel_factor * window_size) across all stages.

        For swinv2_base_window12to24:
          stage 0: factor=4, ws=24 → need pixels % 96 == 0
          stage 1: factor=8, ws=24 → need pixels % 192 == 0
          stage 2: factor=16, ws=24 → need pixels % 384 == 0
          stage 3: factor=32, ws=12 → need pixels % 384 == 0
          LCM = 384
        """
        divisor = 1
        for si in stage_info:
            ws = si['window_size']
            ws_max = max(ws) if isinstance(ws, (tuple, list)) else ws
            needed = si['pixel_factor'] * ws_max
            divisor = _lcm(divisor, needed)
        return divisor

    def _update_swin_resolutions(self, H_pixels, W_pixels, device):
        """
        Update input_resolution AND attn_mask on all Swin-V2 stages and blocks
        to match the actual padded input size.
        """
        for si in self._stage_info:
            stage = si['stage']
            factor = si['pixel_factor']
            block_h = H_pixels // factor
            block_w = W_pixels // factor

            # Update each block's input_resolution and attn_mask
            for block in stage.blocks:
                old_res = block.input_resolution
                new_res = (block_h, block_w)
                block.input_resolution = new_res

                # Recompute attn_mask if resolution changed and block uses shift
                if old_res != new_res and any(block.shift_size):
                    block.attn_mask = _compute_swin_attn_mask(
                        block_h, block_w, block.window_size, block.shift_size, device
                    )
                elif not any(block.shift_size):
                    block.attn_mask = None

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor with tensors (B, 3, H, W) and mask (B, H, W)
        Returns:
            Dict[str, NestedTensor] — one entry per feature level
        """
        x = tensor_list.tensors
        mask = tensor_list.mask

        # Pad input so H, W are divisible by pixel_divisor.
        # This ensures the feature grid at EVERY Swin stage is divisible
        # by that stage's window_size.
        _, _, H, W = x.shape
        div = self.pixel_divisor
        pad_h = (div - H % div) % div
        pad_w = (div - W % div) % div
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right and bottom

        # For Swin-V2: update input_resolution and attn_mask on all blocks
        if self.is_swin:
            self._update_swin_resolutions(x.shape[2], x.shape[3], x.device)

        # timm features_only forward returns a list of feature maps
        features = self.body(x)

        out = {}
        for i, feat in enumerate(features):
            # timm Swin-V2 outputs (B, H, W, C) — convert to (B, C, H, W).
            # Detect NHWC: channel dim (last) should match expected channels,
            # and dim 1 should be spatial (much larger than channel count).
            expected_c = self.num_channels[i]
            if feat.ndim == 4 and feat.shape[-1] == expected_c and feat.shape[1] != expected_c:
                feat = feat.permute(0, 3, 1, 2).contiguous()

            # Use the ORIGINAL (unpadded) mask to create per-level masks.
            feat_mask = F.interpolate(
                mask[None].float(), size=feat.shape[-2:]
            ).to(torch.bool)[0]
            out[str(i)] = NestedTensor(feat, feat_mask)

        return out
