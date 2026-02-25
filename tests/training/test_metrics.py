# tests/training/test_metrics.py
import torch
from training.metrics import psnr, ssim


def test_psnr_identical_images_returns_inf():
    img = torch.rand(3, 64, 64) * 2 - 1  # [-1, 1]
    assert psnr(img, img) == float('inf')


def test_psnr_different_images_returns_positive_float():
    pred = torch.zeros(3, 64, 64)
    target = torch.ones(3, 64, 64)
    result = psnr(pred, target)
    assert isinstance(result, float)
    assert 0 < result < 100


def test_ssim_identical_images_returns_one():
    img = torch.rand(3, 64, 64) * 2 - 1
    result = ssim(img, img)
    assert abs(result - 1.0) < 1e-3


def test_ssim_different_images_returns_float_in_range():
    pred = torch.zeros(3, 64, 64)
    target = torch.ones(3, 64, 64)
    result = ssim(pred, target)
    assert isinstance(result, float)
    assert 0 <= result <= 1
