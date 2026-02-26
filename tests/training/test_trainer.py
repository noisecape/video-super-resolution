# tests/training/test_trainer.py
import torch
from training.train import Trainer

CONFIG = {
    'device': 'cpu',
    'lr': 1e-4,
    'num_epochs': 1,
    'log_interval': 10,
    'checkpoint_dir': '/tmp/test_checkpoints',
    'num_timesteps': 1000,
}

SMALL_CONFIG = {
    **CONFIG,
    'num_timesteps': 2,   # tiny loop so test is fast
    'val_interval': 1,
    'val_num_samples': 1,
}

def test_vae_params_are_frozen():
    trainer = Trainer(CONFIG)
    for param in trainer.vae.parameters():
        assert not param.requires_grad

def test_unet_params_are_trainable():
    trainer = Trainer(CONFIG)
    trainable = [p for p in trainer.unet.parameters() if p.requires_grad]
    assert len(trainable) > 0

def test_train_step_returns_scalar():
    trainer = Trainer(CONFIG)
    target_lr = torch.randn(2, 3, 32, 32)   # half the HR resolution
    target_hr = torch.randn(2, 3, 64, 64)
    loss = trainer.train_step(target_lr, target_hr)
    assert isinstance(loss, float)
    assert loss > 0


def test_train_step_does_not_update_vae():
    trainer = Trainer(CONFIG)
    vae_params_before = [p.clone() for p in trainer.vae.parameters()]
    target_lr = torch.randn(2, 3, 32, 32)
    target_hr = torch.randn(2, 3, 64, 64)
    trainer.train_step(target_lr, target_hr)
    for before, after in zip(vae_params_before, trainer.vae.parameters()):
        assert torch.equal(before, after)


def test_train_step_updates_unet():
    trainer = Trainer(CONFIG)
    unet_params_before = [p.clone() for p in trainer.unet.parameters()]
    target_lr = torch.randn(2, 3, 32, 32)
    target_hr = torch.randn(2, 3, 64, 64)
    trainer.train_step(target_lr, target_hr)
    any_changed = any(
        not torch.equal(before, after)
        for before, after in zip(unet_params_before, trainer.unet.parameters())
    )
    assert any_changed

def test_train_epoch_returns_float():
    trainer = Trainer(CONFIG)
    fake_loader = [
        (torch.randn(2, 3, 32, 32), torch.randn(2, 3, 64, 64))
        for _ in range(3)
    ]
    avg_loss = trainer.train_epoch(fake_loader)
    assert isinstance(avg_loss, float)
    assert avg_loss > 0


def test_train_epoch_handles_full_tuple_batch():
    trainer = Trainer(CONFIG)
    # Full dataset tuple: (context_lr, target_lr, target_hr)
    fake_loader = [
        (torch.randn(2, 6, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.randn(2, 3, 64, 64))
        for _ in range(2)
    ]
    avg_loss = trainer.train_epoch(fake_loader)
    assert isinstance(avg_loss, float)
    assert avg_loss > 0


import os, tempfile

def test_checkpoint_roundtrip():
    trainer = Trainer(CONFIG)
    # Take one step so optimizer state is non-trivial
    target_lr = torch.randn(2, 3, 32, 32)
    target_hr = torch.randn(2, 3, 64, 64)
    trainer.train_step(target_lr, target_hr)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'ckpt.pt')
        trainer.save_checkpoint(path, epoch=1)

        # Load into a fresh trainer and compare UNet params
        trainer2 = Trainer(CONFIG)
        trainer2.load_checkpoint(path)
        for p1, p2 in zip(trainer.unet.parameters(), trainer2.unet.parameters()):
            assert torch.allclose(p1, p2)


def test_validate_metrics_returns_psnr_and_ssim():
    trainer = Trainer(SMALL_CONFIG)
    # (target_lr, target_hr) — LR is half the HR resolution
    fake_loader = [(torch.randn(1, 3, 32, 32), torch.randn(1, 3, 64, 64))]
    metrics = trainer.validate_metrics(fake_loader, num_samples=1)
    assert 'psnr' in metrics and 'ssim' in metrics
    assert isinstance(metrics['psnr'], float)
    assert isinstance(metrics['ssim'], float)


def test_train_calls_validate_epoch():
    config = {**SMALL_CONFIG, 'num_epochs': 1, 'val_interval': 1, 'val_num_samples': 1}
    trainer = Trainer(config)
    train_loader = [(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 64, 64)) for _ in range(2)]
    val_loader = [(torch.randn(1, 3, 32, 32), torch.randn(1, 3, 64, 64))]

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.config['checkpoint_dir'] = tmpdir
        trainer.train(train_loader, val_loader)


def test_validate_epoch_returns_float():
    trainer = Trainer(CONFIG)
    fake_loader = [
        (torch.randn(2, 3, 32, 32), torch.randn(2, 3, 64, 64))
        for _ in range(3)
    ]
    val_loss = trainer.validate_epoch(fake_loader)
    assert isinstance(val_loss, float)
    assert val_loss > 0


def test_validate_epoch_handles_full_tuple_batch():
    trainer = Trainer(CONFIG)
    fake_loader = [
        (torch.randn(2, 6, 3, 32, 32), torch.randn(2, 3, 32, 32), torch.randn(2, 3, 64, 64))
        for _ in range(2)
    ]
    val_loss = trainer.validate_epoch(fake_loader)
    assert isinstance(val_loss, float)
    assert val_loss > 0
