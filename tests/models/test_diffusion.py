# tests/models/test_diffusion.py
import torch
from models.diffusion import NoiseScheduler

def test_add_noise_returns_tuple():
    scheduler = NoiseScheduler()
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    result = scheduler.add_noise(x, t)
    assert isinstance(result, tuple) and len(result) == 2

def test_add_noise_shapes():
    scheduler = NoiseScheduler()
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    x_t, noise = scheduler.add_noise(x, t)
    assert x_t.shape == x.shape
    assert noise.shape == x.shape
