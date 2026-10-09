"""Clinical Evaluation & Uncertainty Quantification on the BraTS Test Set.

Computes:
- Dice Similarity Coefficients (WT, TC, ET)
- 95th Percentile Hausdorff Distance (HD95)
- Monte Carlo Dropout Uncertainty Maps (Entropy & Epistemic Uncertainty)
- Exports predicted NIfTI volumes and uncertainty maps
"""

import argparse
import json
import os
import yaml
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, LoadImaged

from src.data.transforms import get_val_transforms
from src.models.model import build_model
from src.utils.metrics import BraTSMetrics
from src.utils.uncertainty import compute_mc_dropout_uncertainty


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS Evaluation & Uncertainty Analysis")
    parser.add_argument("--checkpoint", type=str, default="./runs/brats_experiment/best_metric_model.pt")
    parser.add_argument("--config", type=str, default="configs/brats_swin_unetr.yaml")
    parser.add_argument("--save-preds", action="store_true", default=True, help="Save NIfTI segmentations & uncertainty maps")
    parser.add_argument("--mc-samples", type=int, default=8, help="Number of MC Dropout passes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load splits
    split_file = cfg["data"]["split_json"]
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}. Run train.py first.")

    with open(split_file, "r") as f:
        splits = json.load(f)
    test_cases = splits["test"]

    print(f"[Evaluation] Found {len(test_cases)} test cases in {split_file}")

    # Build model & load checkpoint
    model = build_model(
        model_name=cfg["model"]["name"],
        img_size=tuple(cfg["model"]["img_size"]),
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        feature_size=cfg["model"]["feature_size"],
    ).to(device)

    if os.path.exists(args.checkpoint):
        print(f"[Evaluation] Loading weights from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"[Warning] Checkpoint '{args.checkpoint}' not found. Evaluating with random initialization for test run.")

    test_transform = Compose([LoadImaged(keys=["image", "label"]), get_val_transforms()])
    metrics_tracker = BraTSMetrics()
    output_eval_dir = os.path.join(cfg["project"]["output_dir"], "evaluation_results")
    os.makedirs(output_eval_dir, exist_ok=True)

    print("=" * 70)
    print(f" Starting Clinical Evaluation on {len(test_cases)} Patients...")
    print(f" Uncertainty Mode: Monte Carlo Dropout ({args.mc_samples} passes)")
    print("=" * 70)

    for idx, case in enumerate(test_cases):
        case_id = case["id"]
        print(f"\n[{idx+1}/{len(test_cases)}] Evaluating Patient: {case_id}")

        data = test_transform(case)
        image = data["image"].unsqueeze(0).to(device)  # [1, 4, H, W, D]
        label = data["label"].unsqueeze(0).to(device)  # [1, 3, H, W, D]

        # Run Monte Carlo Dropout Uncertainty Inference
        results = compute_mc_dropout_uncertainty(
            model=model,
            inputs=image,
            roi_size=tuple(cfg["data"]["spatial_size"]),
            sw_batch_size=cfg["inference"]["sw_batch_size"],
            overlap=cfg["inference"]["overlap"],
            num_samples=args.mc_samples,
            device=args.device,
        )

        binary_pred = results["binary_pred"].to(device)
        metrics_tracker.update(y_pred=binary_pred, y_true=label, compute_hd95=True)

        if args.save_preds:
            patient_save_dir = os.path.join(output_eval_dir, case_id)
            os.makedirs(patient_save_dir, exist_ok=True)
            affine = np.eye(4)

            # Export binary segmentation (3 channels: WT, TC, ET)
            pred_np = binary_pred[0].cpu().numpy().astype(np.uint8)
            nib.save(nib.Nifti1Image(pred_np, affine), os.path.join(patient_save_dir, "pred_subregions.nii.gz"))

            # Export Predictive Entropy (total uncertainty)
            entropy_np = results["entropy_map"][0].numpy().astype(np.float32)
            nib.save(nib.Nifti1Image(entropy_np, affine), os.path.join(patient_save_dir, "entropy_uncertainty.nii.gz"))

            # Export Epistemic Uncertainty (model ambiguity)
            epistemic_np = results["epistemic_map"][0].numpy().astype(np.float32)
            nib.save(nib.Nifti1Image(epistemic_np, affine), os.path.join(patient_save_dir, "epistemic_uncertainty.nii.gz"))

            print(f" -> Saved segmentation and uncertainty maps to {patient_save_dir}")

    # Aggregate cohort results
    final_metrics = metrics_tracker.aggregate(compute_hd95=True)

    print("\n" + "=" * 70)
    print(" FINAL CLINICAL EVALUATION BENCHMARK")
    print("=" * 70)
    print(f" Mean Dice Score:          {final_metrics['dice_mean']:.4f}")
    print(f"  - Whole Tumor (WT):       {final_metrics['dice_wt']:.4f}")
    print(f"  - Tumor Core (TC):        {final_metrics['dice_tc']:.4f}")
    print(f"  - Enhancing Tumor (ET):   {final_metrics['dice_et']:.4f}")
    print("-" * 70)
    if "hd95_mean" in final_metrics:
        print(f" Mean 95% Hausdorff (mm):  {final_metrics['hd95_mean']:.2f}")
        print(f"  - WT HD95:                {final_metrics['hd95_wt']:.2f}")
        print(f"  - TC HD95:                {final_metrics['hd95_tc']:.2f}")
        print(f"  - ET HD95:                {final_metrics['hd95_et']:.2f}")
    print("=" * 70)

    # Save metrics JSON
    with open(os.path.join(output_eval_dir, "test_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    evaluate()
