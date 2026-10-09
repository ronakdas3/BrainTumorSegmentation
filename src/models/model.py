"""Model Factory for 3D Glioma Segmentation.

Includes:
- Swin UNETR (Shifted Window Vision Transformer for 3D volumetric images)
- SegResNet (High efficiency CNN baseline)
- Pretrained self-supervised weight loader
- Test-Time Monte Carlo Dropout enabler for uncertainty estimation
"""

import os
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR


def build_model(
    model_name: str = "SwinUNETR",
    img_size=(128, 128, 128),
    in_channels: int = 4,
    out_channels: int = 3,
    feature_size: int = 48,
    use_checkpoint: bool = True,
    dropout_rate: float = 0.1,
    attn_drop_rate: float = 0.1,
    pretrained_path: str = None,
):
    """Instantiates the 3D model architecture with optional pretrained self-supervised weights."""
    model_name_lower = model_name.lower()

    if model_name_lower == "swinunetr":
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            drop_rate=dropout_rate,
            attn_drop_rate=attn_drop_rate,
            spatial_dims=3,
        )
    elif model_name_lower == "segresnet":
        model = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=dropout_rate,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Use 'SwinUNETR' or 'SegResNet'.")

    # Load self-supervised pretrained backbone if provided
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[Model] Loading self-supervised pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[Model] Checkpoint loaded with status: {msg}")

    return model


def enable_mc_dropout(model: nn.Module):
    """Enables Dropout layers during evaluation mode to allow Monte Carlo Dropout sampling."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
