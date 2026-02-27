# scripts/train.py
import argparse
import yaml
import torch.utils.data as data

from data.vimeo90k import Vimeo90k
from training.train import Trainer


def build_loaders(config: dict):
    train_dataset = Vimeo90k(dataset_mode='train')
    val_dataset = Vimeo90k(dataset_mode='test')

    subset_size = config.get('subset_size')
    if subset_size is not None:
        val_size = min(subset_size // 10, len(val_dataset))
        train_dataset = data.Subset(train_dataset, range(subset_size))
        val_dataset = data.Subset(val_dataset, range(val_size))

    loader_kwargs = dict(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=(config['device'] == 'cuda'),
    )
    train_loader = data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def run(config: dict, train_loader=None, val_loader=None):
    if train_loader is None or val_loader is None:
        train_loader, val_loader = build_loaders(config)
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)


def main():
    parser = argparse.ArgumentParser(description='Train video super-resolution model')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_loader, val_loader = build_loaders(config)
    trainer = Trainer(config)

    if args.resume:
        epoch = trainer.load_checkpoint(args.resume)
        print(f'Resumed from epoch {epoch}')

    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
