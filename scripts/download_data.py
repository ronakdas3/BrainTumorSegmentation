"""Automatic Medical Imaging Dataset Downloader.

Downloads either:
1. MSD Task01 BrainTumour dataset (Medical Segmentation Decathlon) using MONAI's verified mirror.
2. Generates synthetic dummy BraTS scans if offline.
"""

import argparse
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.apps import download_and_extract
from src.data.dataset import create_synthetic_brats_data


def main():
    parser = argparse.ArgumentParser(description="Download medical dataset")
    parser.add_argument("--dataset", type=str, default="msd_brain", choices=["msd_brain", "synthetic"])
    parser.add_argument("--target-dir", type=str, default="./data")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    if args.dataset == "msd_brain":
        print("[Downloader] Downloading MSD Task01 BrainTumour Decathlon dataset...")
        resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
        compressed_file = os.path.join(args.target_dir, "Task01_BrainTumour.tar")
        download_and_extract(
            url=resource,
            filepath=compressed_file,
            output_dir=args.target_dir,
        )
        print(f"[Downloader] Dataset ready at: {os.path.join(args.target_dir, 'Task01_BrainTumour')}")
    else:
        print("[Downloader] Generating synthetic dataset...")
        out = os.path.join(args.target_dir, "synthetic_brats")
        create_synthetic_brats_data(out, num_cases=8)
        print(f"[Downloader] Synthetic dataset ready at: {out}")


if __name__ == "__main__":
    main()
