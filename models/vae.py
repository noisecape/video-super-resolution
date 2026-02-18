# models/vae.py
import torch
import torch.nn as nn
from models.unet import SelfAttention


class ResBlock(nn.Module):
    """
    ResNet block without timestep conditioning.
    Compare to U-Net's ConvBlock: same residual pattern, but no AdaGN.
    The VAE doesn't know about diffusion timesteps.
    """

    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # TODO(human): implement the residual forward pass
        # Hint: norm1 → act → conv1 → norm2 → act → conv2, then add skip(residual)
        # No t_emb argument here — this is the key difference from ConvBlock
        pass


class Encoder(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class VAE(nn.Module):
    pass
