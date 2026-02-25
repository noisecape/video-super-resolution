# training/train.py
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from models.vae import VAE
from models.unet import UNet
from models.diffusion import NoiseScheduler


class Trainer:

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['device'])

        # Models
        self.vae = VAE().to(self.device)
        self.unet = UNet(in_channels=4, out_channels=4).to(self.device)
        self.scheduler = NoiseScheduler(num_timestep=config['num_timesteps'])

        # Move scheduler tensors to device (not nn.Module buffers)
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        # Optimizer (UNet only — VAE has no grad)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=config['lr'])
        self.scaler = GradScaler(enabled=(config['device'] == 'cuda'))

    def train_step(self, batch: torch.Tensor) -> float:
        batch = batch.to(self.device)
        B = batch.shape[0]

        # Encode to latent (frozen VAE, no gradients)
        with torch.no_grad():
            mean, log_var = self.vae.encode(batch)
            z = self.vae.sample(mean, log_var)

        # Sample random timesteps, one per item in batch
        t = torch.randint(0, self.config['num_timesteps'], (B,), device=self.device)

        # Add noise
        x_t, noise = self.scheduler.add_noise(z, t)

        # Predict noise with U-Net, compute loss
        self.optimizer.zero_grad()
        with autocast(enabled=(self.config['device'] == 'cuda')):
            noise_pred = self.unet(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def train_epoch(self, dataloader) -> float:
        self.unet.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            # Dataset may return (context_lr, target_lr, target_hr) or plain tensor
            if isinstance(batch, (list, tuple)):
                batch = batch[2]  # target_hr is the HR ground truth
            loss = self.train_step(batch)
            total_loss += loss
            if (step + 1) % self.config['log_interval'] == 0:
                print(f"  step {step + 1}: loss={loss:.4f}")
        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader) -> float:
        self.unet.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[2]  # target_hr
                batch = batch.to(self.device)
                B = batch.shape[0]
                mean, log_var = self.vae.encode(batch)
                z = self.vae.sample(mean, log_var)
                t = torch.randint(0, self.config['num_timesteps'], (B,), device=self.device)
                x_t, noise = self.scheduler.add_noise(z, t)
                noise_pred = self.unet(x_t, t)
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
        self.unet.train()
        return total_loss / len(dataloader)

    def save_checkpoint(self, path: str, epoch: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(ckpt['unet_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        return ckpt['epoch']

    def train(self, dataloader):
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        for epoch in range(1, self.config['num_epochs'] + 1):
            avg_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch}/{self.config['num_epochs']} — avg loss: {avg_loss:.4f}")
            ckpt_path = os.path.join(
                self.config['checkpoint_dir'], f"checkpoint_epoch{epoch:04d}.pt"
            )
            self.save_checkpoint(ckpt_path, epoch=epoch)
            print(f"Saved checkpoint: {ckpt_path}")


def main():
    import torch.utils.data as data
    from data.vimeo90k import Vimeo90k

    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-4,
        'num_epochs': 100,
        'log_interval': 100,
        'checkpoint_dir': 'checkpoints/',
        'num_timesteps': 1000,
    }

    dataset = Vimeo90k(dataset_mode='train')
    loader = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    trainer = Trainer(config)
    trainer.train(loader)


if __name__ == '__main__':
    main()
