import torch
from scripts.train import run

TINY_CONFIG = {
    'device': 'cpu',
    'lr': 1e-4,
    'num_epochs': 1,
    'batch_size': 2,
    'num_workers': 0,
    'num_timesteps': 2,
    'log_interval': 1,
    'val_interval': 1,
    'val_num_samples': 1,
    'subset_size': None,
}


def test_run_completes_and_saves_checkpoint(tmp_path):
    config = {**TINY_CONFIG, 'checkpoint_dir': str(tmp_path)}
    fake_loader = [
        (torch.randn(2, 3, 32, 32), torch.randn(2, 3, 64, 64))
        for _ in range(2)
    ]
    run(config, train_loader=fake_loader, val_loader=fake_loader)
    checkpoints = list(tmp_path.glob('*.pt'))
    assert len(checkpoints) == 1
