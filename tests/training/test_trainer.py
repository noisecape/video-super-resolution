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
