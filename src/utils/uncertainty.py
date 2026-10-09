"""Monte Carlo Dropout Uncertainty Estimation for 3D Medical Segmentation.

Implements Bayesian Deep Learning approximation via test-time dropout:
- Draws T stochastic forward passes
- Computes Predictive Mean, Predictive Variance, Predictive Entropy
- Computes Epistemic Uncertainty (Mutual Information) separating model ambiguity
  from scan noise.
"""

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from src.models.model import enable_mc_dropout


def compute_mc_dropout_uncertainty(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    roi_size=(128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    num_samples: int = 10,
    device: str = "cuda",
):
    """Executes Monte Carlo Dropout inference over a 3D medical volume.

    Args:
        model: SwinUNETR or SegResNet model
        inputs: Input 3D multi-contrast volume [1, 4, H, W, D]
        roi_size: Sliding window patch size
        sw_batch_size: Sliding window batch size
        overlap: Overlap between adjacent sliding window patches
        num_samples: Number of stochastic forward passes (T)
        device: 'cuda' or 'cpu'

    Returns:
        mean_pred: Sigmoided predictive probability mean [1, 3, H, W, D]
        binary_pred: Thresholded binary segmentation mask (prob >= 0.5) [1, 3, H, W, D]
        variance_map: Predictive variance across MC passes [1, 3, H, W, D]
        entropy_map: Total predictive entropy [1, 3, H, W, D]
        epistemic_map: Mutual information (model uncertainty) [1, 3, H, W, D]
    """
    model.eval()
    enable_mc_dropout(model)  # Re-enable dropout during inference

    inputs = inputs.to(device)
    mc_probs = []

    print(f"[Uncertainty] Running {num_samples} Monte Carlo forward passes with sliding window...")

    with torch.no_grad():
        for i in range(num_samples):
            # Run sliding window inference for the current stochastic pass
            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
            probs = torch.sigmoid(logits)  # Multi-label sub-regions
            mc_probs.append(probs)

    # Stack along sample dimension -> shape: [T, 1, C, H, W, D]
    stacked_probs = torch.stack(mc_probs, dim=0)

    # 1. Predictive Mean: \bar{p} = (1/T) * \sum p_t
    mean_pred = torch.mean(stacked_probs, dim=0)

    # 2. Predictive Variance: \sigma^2 = (1/T) * \sum (p_t - \bar{p})^2
    variance_map = torch.var(stacked_probs, dim=0, unbiased=False)

    # 3. Binary Segmentation Decision: threshold at 0.5
    binary_pred = (mean_pred >= 0.5).to(dtype=torch.float32)

    # 4. Total Predictive Entropy: H(p) = - [p * log(p) + (1-p) * log(1-p)]
    eps = 1e-7
    clamped_mean = torch.clamp(mean_pred, eps, 1.0 - eps)
    entropy_map = -(
        clamped_mean * torch.log(clamped_mean)
        + (1.0 - clamped_mean) * torch.log(1.0 - clamped_mean)
    )

    # 5. Expected Entropy across samples: (1/T) * \sum H(p_t)
    clamped_samples = torch.clamp(stacked_probs, eps, 1.0 - eps)
    sample_entropy = -(
        clamped_samples * torch.log(clamped_samples)
        + (1.0 - clamped_samples) * torch.log(1.0 - clamped_samples)
    )
    expected_entropy = torch.mean(sample_entropy, dim=0)

    # 6. Epistemic Uncertainty (Mutual Information): I = H(mean) - E[H(sample)]
    epistemic_map = torch.clamp(entropy_map - expected_entropy, min=0.0)

    return {
        "mean_pred": mean_pred.cpu(),
        "binary_pred": binary_pred.cpu(),
        "variance_map": variance_map.cpu(),
        "entropy_map": entropy_map.cpu(),
        "epistemic_map": epistemic_map.cpu(),
    }
