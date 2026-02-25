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


def test_step_output_shape():
    scheduler = NoiseScheduler(num_timestep=10)
    x_t = torch.randn(2, 4, 8, 8)
    noise_pred = torch.randn(2, 4, 8, 8)
    x_prev = scheduler.step(x_t, noise_pred, timestep=5)
    assert x_prev.shape == x_t.shape


def test_step_at_t0_is_deterministic():
    scheduler = NoiseScheduler(num_timestep=10)
    x_t = torch.randn(2, 4, 8, 8)
    noise_pred = torch.randn(2, 4, 8, 8)
    out1 = scheduler.step(x_t, noise_pred, timestep=0)
    out2 = scheduler.step(x_t, noise_pred, timestep=0)
    assert torch.equal(out1, out2)


def test_step_at_t_gt0_is_stochastic():
    scheduler = NoiseScheduler(num_timestep=10)
    x_t = torch.randn(2, 4, 8, 8)
    noise_pred = torch.randn(2, 4, 8, 8)
    out1 = scheduler.step(x_t, noise_pred, timestep=5)
    out2 = scheduler.step(x_t, noise_pred, timestep=5)
    assert not torch.equal(out1, out2)
