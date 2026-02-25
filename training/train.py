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
