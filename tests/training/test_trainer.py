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
    batch = torch.randn(2, 3, 64, 64)  # fake HR frames
    loss = trainer.train_step(batch)
    assert isinstance(loss, float)
    assert loss > 0

def test_train_step_does_not_update_vae():
    trainer = Trainer(CONFIG)
    vae_params_before = [p.clone() for p in trainer.vae.parameters()]
    batch = torch.randn(2, 3, 64, 64)
    trainer.train_step(batch)
    for before, after in zip(vae_params_before, trainer.vae.parameters()):
        assert torch.equal(before, after)

def test_train_step_updates_unet():
    trainer = Trainer(CONFIG)
    unet_params_before = [p.clone() for p in trainer.unet.parameters()]
    batch = torch.randn(2, 3, 64, 64)
    trainer.train_step(batch)
    any_changed = any(
        not torch.equal(before, after)
        for before, after in zip(unet_params_before, trainer.unet.parameters())
    )
    assert any_changed

def test_train_epoch_returns_float():
    trainer = Trainer(CONFIG)
    # Fake dataloader: list of 3 batches
    fake_loader = [torch.randn(2, 3, 64, 64) for _ in range(3)]
    avg_loss = trainer.train_epoch(fake_loader)
    assert isinstance(avg_loss, float)
    assert avg_loss > 0
