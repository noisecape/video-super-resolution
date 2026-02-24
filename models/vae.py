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
        # norm1 → act → conv1 → norm2 → act → conv2, then add skip(residual)
        residual = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        x = x + self.skip(residual)
        
        return x


class Encoder(nn.Module):
    """
    Encodes RGB images → latent distribution params (mean + log_var).
    SD KL-f8 spec: 3 downsamples (8x), 4ch latent, attention at bottleneck only.
    """

    def __init__(self, in_channels=3, latent_channels=4,
                 base_channels=128, channel_mults=(1, 2, 4, 4), num_groups=32):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.levels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch_in = base_channels
        for i, mult in enumerate(channel_mults):
            ch_out = base_channels * mult
            self.levels.append(nn.ModuleList([
                ResBlock(ch_in, ch_out, num_groups),
                ResBlock(ch_out, ch_out, num_groups),
            ]))
            # Downsample between all levels except the last
            if i < len(channel_mults) - 1:
                self.downsamplers.append(
                    nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1)
                )
            else:
                self.downsamplers.append(None)
            ch_in = ch_out

        # Mid block: ResBlock → Attention → ResBlock (at lowest resolution)
        mid_ch = base_channels * channel_mults[-1]
        self.mid_res1 = ResBlock(mid_ch, mid_ch, num_groups)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, num_groups)

        # Output conv: projects to 2 * latent_channels (mean + log_var stacked)
        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_groups, mid_ch),
            nn.SiLU(),
            nn.Conv2d(mid_ch, latent_channels * 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Steps:
        #   1. Apply self.input_conv
        x = self.input_conv(x)
        #   2. For each (level, downsampler) in zip(self.levels, self.downsamplers):
        for (level, downsampler) in zip(self.levels, self.downsamplers):
        # pass x through level[0] and level[1] (ResBlocks)
            x = level[0](x)
            x = level[1](x)
        # if downsampler is not None, apply it
            if downsampler != None:
                x = downsampler(x)
        #   3. Apply mid block: mid_res1 → mid_attn → mid_res2
        x = self.mid_res1(x)
        x = self.mid_attn(x)
        x = self.mid_res2(x)
        #   4. Apply output_conv and return
        x = self.output_conv(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_channels=4, out_channels=3,
                 base_channels=128, channel_mults=(1, 2, 4, 4), num_groups=32):
        super().__init__()

        mid_ch = base_channels * channel_mults[-1]
        self.input_conv = nn.Conv2d(in_channels, mid_ch, kernel_size=3, padding=1)

        # Mid block: ResBlock → Attention → ResBlock
        self.mid_res1 = ResBlock(mid_ch, mid_ch, num_groups)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, num_groups)

        # Decoder levels with upsampling (reverse of encoder)
        self.levels = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        ch_in = mid_ch
        for i, mult in enumerate(reversed(channel_mults)):
            ch_out = base_channels * mult
            self.levels.append(nn.ModuleList([
                ResBlock(ch_in, ch_out, num_groups),
                ResBlock(ch_out, ch_out, num_groups),
            ]))
            if i < len(channel_mults) - 1:
                self.upsamplers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
                ))
            else:
                self.upsamplers.append(None)
            ch_in = ch_out

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_groups, ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.mid_res1(x)
        x = self.mid_attn(x)
        x = self.mid_res2(x)
        for level, up in zip(self.levels, self.upsamplers):
            x = level[0](x)
            x = level[1](x)
            if up is not None:
                x = up(x)
        return self.output_conv(x)

class VAE(nn.Module):
    """
    Full VAE: Encoder + reparameterisation + Decoder.
    Matches SD KL-f8 architecture for pretrained weight loading.
    """

    SCALE_FACTOR = 0.18215  # SD empirical latent normalisation constant

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        """RGB image → (mean, log_var), each (B, 4, H/8, W/8)."""
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=1)
        log_var = torch.clamp(log_var, -30.0, 20.0)  # numerical stability
        return mean, log_var

    def sample(self, mean, log_var):
        """
        Returns a scaled latent z ready for the diffusion U-Net.

        Steps:
          1. Compute std from log_var:  std = exp(0.5 * log_var)
          2. Sample epsilon ~ N(0, I) using torch.randn_like(std)
          3. z = mean + std * epsilon
          4. Return z * SCALE_FACTOR
        """
        std = torch.exp(0.5 * log_var) # short for: exp(0.5 · log_var) = exp(log_var)^0.5 = sqrt(exp(log_var))
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon # reparametrisation trick!

        return z * self.SCALE_FACTOR

    def decode(self, z):
        """Scaled latent → RGB image."""
        return self.decoder(z / self.SCALE_FACTOR)

    def forward(self, x):
        """Full encode → sample → decode for VAE training."""
        mean, log_var = self.encode(x)
        z = self.sample(mean, log_var)
        recon = self.decode(z)
        return recon, mean, log_var

    def kl_loss(self, mean, log_var):
        """
        Implement KL divergence between N(mean, var) and N(0, I).

        Formula: -0.5 * mean_over_all_elements(1 + log_var - mean² - exp(log_var))

        Hints:
          - use mean.pow(2) for mean²
          - use log_var.exp() for exp(log_var)
          - use torch.mean() to reduce to a scalar

        Sanity check: when mean=0 and log_var=0 → result should be exactly 0.
        """
        loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        return loss
 