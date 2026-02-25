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
