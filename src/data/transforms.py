"""Data preprocessing and augmentation pipelines for 3D Multimodal BraTS MRI.

Converts multi-sequence MRI (FLAIR, T1, T1ce, T2) and multiclass segmentation labels into
3 clinical evaluation sub-regions:
  - Channel 0: Whole Tumor (WT) = NCR + ED + ET (labels 1, 2, 4)
  - Channel 1: Tumor Core (TC) = NCR + ET (labels 1, 4)
  - Channel 2: Enhancing Tumor (ET) = ET (label 4)
"""

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    Spacingd,
)


class ConvertToBratsSubregionsd(MapTransform):
    """Convert integer BraTS labels into 3 overlapping binary sub-regions (WT, TC, ET).

    BraTS standard labels:
      - 1: Necrotic / Non-enhancing tumor core (NCR)
      - 2: Peritumoral edema (ED)
      - 4: Enhancing tumor (ET) (sometimes label 3 in newer formats)
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            # Squeeze to 3D if channel-first (1, H, W, D)
            if label.ndim == 4 and label.shape[0] == 1:
                label_3d = label[0]
            else:
                label_3d = label

            # Region 1: Whole Tumor (WT) = NCR (1) | ED (2) | ET (4 or 3)
            wt = torch.logical_or(
                torch.logical_or(label_3d == 1, label_3d == 2),
                torch.logical_or(label_3d == 4, label_3d == 3),
            )

            # Region 2: Tumor Core (TC) = NCR (1) | ET (4 or 3)
            tc = torch.logical_or(
                label_3d == 1,
                torch.logical_or(label_3d == 4, label_3d == 3),
            )

            # Region 3: Enhancing Tumor (ET) = ET (4 or 3)
            et = torch.logical_or(label_3d == 4, label_3d == 3)

            # Stack along channel dimension -> shape (3, H, W, D)
            subregions = torch.stack([wt, tc, et], dim=0).to(dtype=torch.float32)
            d[key] = subregions
        return d


def get_train_transforms(spatial_size=(128, 128, 128), pos_sample_num=1, neg_sample_num=1):
    """Build the training transform pipeline with 3D spatial and intensity augmentations."""
    return Compose(
        [
            EnsureTyped(keys=["image", "label"], track_meta=False),
            ConvertToBratsSubregionsd(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Oversample tumor regions to handle extreme sparsity of lesions in 3D volumes
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=pos_sample_num,
                neg=neg_sample_num,
                num_samples=pos_sample_num + neg_sample_num,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.2),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms():
    """Deterministic validation transform pipeline."""
    return Compose(
        [
            EnsureTyped(keys=["image", "label"], track_meta=False),
            ConvertToBratsSubregionsd(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_test_transforms():
    """Inference transforms for unseen test cases (labels optional)."""
    return Compose(
        [
            EnsureTyped(keys=["image"], track_meta=False),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )
