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
    
    def __init__(
        self,
        in_channels=4, out_channels=3,
        base_channels=128, channel_mults=(1, 2, 4, 4), num_groups=32
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, base_channels*channel_mults[-1], kernel_size=3, padding=1)
        self.levels = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
            
        # Mid block: ResBlock → Attention → ResBlock (at lowest resolution)
        mid_ch = base_channels * channel_mults[-1]
        self.mid_res1 = ResBlock(mid_ch, mid_ch, num_groups)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, num_groups)

        # upsample
        ch_in = base_channels * channel_mults[-1]
        for i, mult in enumerate(reversed(channel_mults)):
            ch_out = base_channels * mult
            self.levels.append(nn.ModuleList([
                ResBlock(in_channels=ch_in, out_channels=ch_out, num_groups=num_groups),
                ResBlock(in_channels=ch_out, out_channels=ch_out, num_groups=num_groups)
            ]))

            if i < len(channel_mults) - 1:
                self.upsamplers.append(
                    nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
                    )
            else:
                self.upsamplers.append(None)
            ch_in = ch_out

        self.output_conv = nn.Conv2d(base_channels, out_channels=out_channels, kernel_size=3, padding=1)
            
    def forward(self, x):

        # 1. input conv
        x = self.input_conv(x)
        # 2. Apply mid block
        x = self.mid_res1(x)
        x = self.mid_attn(x)
        x = self.mid_res2(x)
        # 3. iterate through levels to upscale
        for (levels, upsamplers) in zip(self.levels, self.upsamplers):
            x = levels[0](x)
            x = levels[1](x)
            if upsamplers != None:
                x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = upsamplers(x)

        x = self.output_conv(x)

        return x

class VAE(nn.Module):
    pass
