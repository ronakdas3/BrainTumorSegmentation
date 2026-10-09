"""BraTS Dataset Loader and Split Generator.

Supports:
- BraTS 2020 / 2021 / 2023 multi-file folders (flair, t1, t1ce, t2, seg)
- MSD Task01_BrainTumour 4D NIfTI format
- Synthetic dummy volume generator for rapid dry-runs and pipeline testing
"""

import json
import os
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from sklearn.model_selection import train_test_split
from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
from monai.transforms import LoadImaged, Compose


def find_brats_cases(data_dir: str):
    """Scans data_dir and returns a list of dictionaries with paths to MRI modalities."""
    data_path = Path(data_dir)
    cases = []

    # Check for BraTS multi-sequence per-folder structure
    patient_dirs = [p for p in data_path.iterdir() if p.is_dir() and not p.name.startswith(".")]

    for pdir in sorted(patient_dirs):
        pname = pdir.name
        flair = list(pdir.glob("*_flair.nii*")) or list(pdir.glob("*flair*.nii*"))
        t1 = list(pdir.glob("*_t1.nii*")) or list(pdir.glob("*t1*.nii*"))
        # Exclude t1ce from standard t1
        t1 = [f for f in t1 if "t1ce" not in f.name.lower() and "t1gd" not in f.name.lower()]
        t1ce = list(pdir.glob("*_t1ce.nii*")) or list(pdir.glob("*t1gd*.nii*")) or list(pdir.glob("*t1ce*.nii*"))
        t2 = list(pdir.glob("*_t2.nii*")) or list(pdir.glob("*t2*.nii*"))
        seg = list(pdir.glob("*_seg.nii*")) or list(pdir.glob("*seg*.nii*"))

        if flair and t1 and t1ce and t2:
            case_dict = {
                "id": pname,
                "image": [str(flair[0]), str(t1[0]), str(t1ce[0]), str(t2[0])],
            }
            if seg:
                case_dict["label"] = str(seg[0])
            cases.append(case_dict)

    # Check for MSD Task01 style (single 4D volume per patient)
    if not cases:
        img_files = sorted(list(data_path.glob("imagesTr/*.nii*")))
        for img in img_files:
            pname = img.name.replace(".nii.gz", "").replace(".nii", "")
            seg_candidates = list(data_path.glob(f"labelsTr/{img.name}"))
            case_dict = {
                "id": pname,
                "image": str(img),
            }
            if seg_candidates:
                case_dict["label"] = str(seg_candidates[0])
            cases.append(case_dict)

    return cases


def get_brats_splits(data_dir: str, split_json: str, ratios=(0.8, 0.1, 0.1), seed: int = 42):
    """Generates patient-level stratified train / val / test splits to avoid data leakage."""
    if os.path.exists(split_json):
        with open(split_json, "r") as f:
            splits = json.load(f)
        print(f"[Dataset] Loaded existing patient splits from: {split_json}")
        return splits["train"], splits["val"], splits["test"]

    cases = find_brats_cases(data_dir)
    if not cases:
        raise ValueError(
            f"No BraTS cases found in '{data_dir}'. Ensure folder contains patient directories "
            f"with *_flair.nii.gz, *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_seg.nii.gz."
        )

    r_train, r_val, r_test = ratios
    train_cases, temp_cases = train_test_split(cases, test_size=(1.0 - r_train), random_state=seed)
    val_weight = r_val / (r_val + r_test)
    val_cases, test_cases = train_test_split(temp_cases, test_size=(1.0 - val_weight), random_state=seed)

    splits = {
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
    }

    os.makedirs(os.path.dirname(os.path.abspath(split_json)), exist_ok=True)
    with open(split_json, "w") as f:
        json.dump(splits, f, indent=2)

    print(
        f"[Dataset] Generated patient-level splits -> Train: {len(train_cases)}, "
        f"Val: {len(val_cases)}, Test: {len(test_cases)}. Saved to {split_json}"
    )
    return train_cases, val_cases, test_cases


def get_dataloaders(
    train_files,
    val_files,
    train_transform,
    val_transform,
    batch_size: int = 2,
    num_workers: int = 4,
    cache_rate: float = 1.0,
):
    """Creates MONAI CacheDataLoaders for high-throughput GPU training."""
    # Prepend reader transform so file paths get converted to tensors
    train_transform_full = Compose([LoadImaged(keys=["image", "label"]), train_transform])
    val_transform_full = Compose([LoadImaged(keys=["image", "label"]), val_transform])

    if cache_rate > 0.0:
        print(f"[DataLoader] Using CacheDataset with cache_rate={cache_rate}")
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transform_full,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transform_full,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        train_ds = Dataset(data=train_files, transform=train_transform_full)
        val_ds = Dataset(data=val_files, transform=val_transform_full)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def create_synthetic_brats_data(output_dir: str, num_cases: int = 4, shape=(128, 128, 128)):
    """Generates synthetic 3D multi-contrast MRI volumes and BraTS masks for pipeline verification."""
    os.makedirs(output_dir, exist_ok=True)
    affine = np.eye(4)

    for i in range(num_cases):
        case_id = f"BraTS2021_SYNTH_{i:04d}"
        case_dir = Path(output_dir) / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic brain volume (ellipsoid)
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        brain_mask = ((x - cx) ** 2 / (cx * 0.8) ** 2 +
                      (y - cy) ** 2 / (cy * 0.8) ** 2 +
                      (z - cz) ** 2 / (cz * 0.8) ** 2) <= 1.0

        # Create tumor core and surrounding edema
        tx, ty, tz = cx + 15, cy - 10, cz + 5
        edema_mask = (((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2) <= 22 ** 2) & brain_mask
        core_mask = (((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2) <= 14 ** 2) & brain_mask
        enhancing_mask = (((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2) <= 7 ** 2) & brain_mask

        # Multi-label segmentation: 0=bg, 1=NCR, 2=ED, 4=ET
        seg = np.zeros(shape, dtype=np.uint8)
        seg[edema_mask] = 2
        seg[core_mask] = 1
        seg[enhancing_mask] = 4

        # Generate 4 modalities with distinct intensity characteristics
        flair = np.random.normal(0.2, 0.05, shape) * brain_mask
        flair[edema_mask] += 0.8
        flair[core_mask] += 0.4

        t1 = np.random.normal(0.5, 0.05, shape) * brain_mask
        t1[core_mask] -= 0.3

        t1ce = t1.copy()
        t1ce[enhancing_mask] += 0.9

        t2 = np.random.normal(0.3, 0.05, shape) * brain_mask
        t2[edema_mask] += 0.7

        # Save NIfTI files
        nib.save(nib.Nifti1Image(flair.astype(np.float32), affine), case_dir / f"{case_id}_flair.nii.gz")
        nib.save(nib.Nifti1Image(t1.astype(np.float32), affine), case_dir / f"{case_id}_t1.nii.gz")
        nib.save(nib.Nifti1Image(t1ce.astype(np.float32), affine), case_dir / f"{case_id}_t1ce.nii.gz")
        nib.save(nib.Nifti1Image(t2.astype(np.float32), affine), case_dir / f"{case_id}_t2.nii.gz")
        nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine), case_dir / f"{case_id}_seg.nii.gz")

    print(f"[Dataset] Generated {num_cases} synthetic BraTS cases in: {output_dir}")
