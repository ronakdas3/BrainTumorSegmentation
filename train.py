"""Training pipeline for 3D BraTS Brain Tumor Segmentation.

Features:
- Swin UNETR / SegResNet with mixed precision (AMP)
- DiceFocalLoss for severe class imbalance in 3D volumes
- Sliding window validation
- Automatic patient split management
- Synthetic mock data generation for instant pipeline verification
"""

import argparse
import os
import time
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.utils import set_determinism

from src.data.dataset import create_synthetic_brats_data, get_brats_splits, get_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model import build_model
from src.utils.metrics import BraTSMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS 3D Segmentation Training")
    parser.add_argument("--config", type=str, default="configs/brats_swin_unetr.yaml", help="Config file path")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic data & run quick dry-run test")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg["train"]["max_epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    device = torch.device(args.device)
    set_determinism(seed=cfg["project"]["seed"])
    output_dir = cfg["project"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    print("=" * 70)
    print(f" BraTS 3D Segmentation Training [{cfg['model']['name']}]")
    print(f" Device: {device} | Mixed Precision: {cfg['train']['amp']}")
    print("=" * 70)

    # Handle synthetic test data mode
    data_dir = cfg["data"]["data_dir"]
    if args.mock or not os.path.exists(data_dir):
        if not os.path.exists(data_dir):
            print(f"[Info] Real data directory '{data_dir}' not found.")
        print("[Info] Setting up synthetic BraTS dataset for pipeline execution...")
        data_dir = "./data/synthetic_brats"
        if args.mock:
            cfg["model"]["feature_size"] = min(cfg["model"]["feature_size"], 12)
            cfg["data"]["spatial_size"] = [64, 64, 64]
            cfg["inference"]["sw_batch_size"] = 1
            cfg["data"]["cache_rate"] = 0.0
            create_synthetic_brats_data(data_dir, num_cases=4, shape=(64, 64, 64))
        else:
            create_synthetic_brats_data(data_dir, num_cases=6, shape=tuple(cfg["data"]["spatial_size"]))
        cfg["data"]["split_json"] = "./data/synthetic_splits.json"
        if os.path.exists(cfg["data"]["split_json"]):
            os.remove(cfg["data"]["split_json"])

    # Load splits & dataloaders
    train_files, val_files, test_files = get_brats_splits(
        data_dir=data_dir,
        split_json=cfg["data"]["split_json"],
        ratios=tuple(cfg["data"]["train_val_test_ratio"]),
        seed=cfg["project"]["seed"],
    )

    train_trans = get_train_transforms(
        spatial_size=tuple(cfg["data"]["spatial_size"]),
        pos_sample_num=cfg["data"]["pos_sample_num"],
        neg_sample_num=cfg["data"]["neg_sample_num"],
    )
    val_trans = get_val_transforms()

    train_loader, val_loader = get_dataloaders(
        train_files=train_files,
        val_files=val_files,
        train_transform=train_trans,
        val_transform=val_trans,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"] if device.type == "cuda" else 0,
        cache_rate=cfg["data"]["cache_rate"] if device.type == "cuda" else 0.0,
    )

    # Initialize model
    model = build_model(
        model_name=cfg["model"]["name"],
        img_size=tuple(cfg["model"]["img_size"]),
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        feature_size=cfg["model"]["feature_size"],
        use_checkpoint=cfg["model"]["use_checkpoint"],
        dropout_rate=cfg["model"]["dropout_rate"],
        attn_drop_rate=cfg["model"]["attn_drop_rate"],
        pretrained_path=cfg["model"].get("pretrained_weights"),
    ).to(device)

    # Loss function (DiceFocalLoss handles extreme foreground/background imbalance)
    if cfg["train"]["loss_type"] == "DiceFocalLoss":
        loss_fn = DiceFocalLoss(sigmoid=True, lambda_focal=1.0)
    else:
        loss_fn = DiceCELoss(sigmoid=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["max_epochs"],
        eta_min=float(cfg["train"]["min_lr"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")
    metrics_tracker = BraTSMetrics()

    best_val_dice = 0.0
    total_epochs = cfg["train"]["max_epochs"]
    val_interval = cfg["train"]["val_interval"]

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        t0 = time.time()

        for batch_data in train_loader:
            step_count += 1
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda"):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            if cfg["train"]["amp"] and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip_val"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip_val"])
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= max(step_count, 1)
        writer.add_scalar("Train/Loss", epoch_loss, epoch)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
        print(f"Epoch [{epoch:03d}/{total_epochs:03d}] Loss: {epoch_loss:.4f} | Time: {time.time()-t0:.1f}s")

        # Validation step
        if epoch % val_interval == 0 or epoch == total_epochs:
            model.eval()
            metrics_tracker.reset()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for val_data in val_loader:
                    val_steps += 1
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    with torch.amp.autocast("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda"):
                        val_outputs = sliding_window_inference(
                            inputs=val_images,
                            roi_size=tuple(cfg["data"]["spatial_size"]),
                            sw_batch_size=cfg["inference"]["sw_batch_size"],
                            predictor=model,
                            overlap=cfg["inference"]["overlap"],
                            mode="gaussian",
                        )
                        v_loss = loss_fn(val_outputs, val_labels)
                        val_loss += v_loss.item()

                    # Binarize prediction for Dice metric
                    pred_binary = (torch.sigmoid(val_outputs) >= 0.5).to(dtype=torch.float32)
                    metrics_tracker.update(y_pred=pred_binary, y_true=val_labels)

            val_metrics = metrics_tracker.aggregate()
            mean_dice = val_metrics["dice_mean"]
            val_loss /= max(val_steps, 1)

            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/Dice_Mean", mean_dice, epoch)
            writer.add_scalar("Val/Dice_WT", val_metrics["dice_wt"], epoch)
            writer.add_scalar("Val/Dice_TC", val_metrics["dice_tc"], epoch)
            writer.add_scalar("Val/Dice_ET", val_metrics["dice_et"], epoch)

            print(
                f" -> Validation: Dice Mean: {mean_dice:.4f} | "
                f"WT: {val_metrics['dice_wt']:.4f}, TC: {val_metrics['dice_tc']:.4f}, ET: {val_metrics['dice_et']:.4f}"
            )

            # Save best checkpoint
            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                ckpt_path = os.path.join(output_dir, "best_metric_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "best_dice": best_val_dice,
                        "config": cfg,
                    },
                    ckpt_path,
                )
                print(f" *** New Best Model saved! Mean Dice: {best_val_dice:.4f} -> {ckpt_path} ***")

        # Always save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "config": cfg,
            },
            os.path.join(output_dir, "latest_model.pt"),
        )

    writer.close()
    print("=" * 70)
    print(f"Training completed. Best Validation Mean Dice: {best_val_dice:.4f}")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    train()
