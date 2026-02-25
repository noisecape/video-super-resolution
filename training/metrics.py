# training/metrics.py
import torch
import numpy as np
from skimage.metrics import structural_similarity


def _to_numpy_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float32 tensor in [-1, 1] → (H, W, C) uint8 numpy array.

    Uses tolist() to avoid the broken torch↔NumPy C bridge (PyTorch 2.1 + NumPy 2.x).
    """
    arr = np.array(((t.cpu().clamp(-1, 1) + 1) * 127.5).tolist(), dtype=np.uint8)
    return arr.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two (C, H, W) float32 tensors in [-1, 1]."""
    pred_np = _to_numpy_uint8_hwc(pred).astype(np.float32)
    target_np = _to_numpy_uint8_hwc(target).astype(np.float32)
    mse = np.mean((pred_np - target_np) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(255.0 ** 2 / mse))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM between two (C, H, W) float32 tensors in [-1, 1]."""
    pred_np = _to_numpy_uint8_hwc(pred)
    target_np = _to_numpy_uint8_hwc(target)
    return float(structural_similarity(pred_np, target_np, channel_axis=2, data_range=255))
