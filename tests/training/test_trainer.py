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

def test_train_epoch_handles_tuple_batch():
    trainer = Trainer(CONFIG)
    # Simulate dataset that returns (context_lr, target_lr, target_hr)
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
    batch = torch.randn(2, 3, 64, 64)
    trainer.train_step(batch)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'ckpt.pt')
        trainer.save_checkpoint(path, epoch=1)

        # Load into a fresh trainer and compare UNet params
        trainer2 = Trainer(CONFIG)
        trainer2.load_checkpoint(path)
        for p1, p2 in zip(trainer.unet.parameters(), trainer2.unet.parameters()):
            assert torch.allclose(p1, p2)
