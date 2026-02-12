import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self, in_channels=8, out_channels=4, base_channels=64,
                 channel_mults=(1, 2, 4, 8), time_emb_dim=256, num_groups=32):
        super().__init__()

        # --- Timestep embedding: sinusoidal → MLP projection ---
        # Sinusoidal gives us (B, base_channels), MLP projects to (B, time_emb_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- Initial convolution: raw input → base_channels ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # --- Encoder: progressively downsample ---
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        channels = [base_channels]  # track channel sizes for skip connections
        ch_in = base_channels
        for mult in channel_mults[:-1]:  # exclude last mult (that's the bottleneck)
            ch_out = base_channels * mult
            self.encoder_blocks.append(nn.ModuleList([
                ConvBlock(ch_in, ch_out, time_emb_dim, num_groups),
                ConvBlock(ch_out, ch_out, time_emb_dim, num_groups),
            ]))
            self.downsamplers.append(Downsample(ch_out))
            channels.append(ch_out)
            ch_in = ch_out

        # --- Bottleneck ---
        bottleneck_ch = base_channels * channel_mults[-1]
        self.bottleneck = nn.ModuleList([
            ConvBlock(ch_in, bottleneck_ch, time_emb_dim, num_groups),
            ConvBlock(bottleneck_ch, bottleneck_ch, time_emb_dim, num_groups),
        ])

        # --- Decoder: progressively upsample ---
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        ch_in = bottleneck_ch
        for mult in reversed(channel_mults[:-1]):
            ch_out = base_channels * mult
            ch_skip = channels.pop()  # matching encoder's output channels
            # ch_in + ch_skip because we concatenate skip connections
            self.decoder_blocks.append(nn.ModuleList([
                ConvBlock(ch_in + ch_skip, ch_out, time_emb_dim, num_groups),
                ConvBlock(ch_out, ch_out, time_emb_dim, num_groups),
            ]))
            self.upsamplers.append(Upsample(ch_in))
            ch_in = ch_out

        # --- Final output projection ---
        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_groups, ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, out_channels, kernel_size=1),
        )

    def forward(self, x, t):
        """
        Args:
            x: Noisy input tensor (B, in_channels, H, W)
            t: Timestep tensor (B,)
        Returns:
            Predicted noise (B, out_channels, H, W)
        """
        # Step 1: Compute timestep embedding using self.time_embed(t)
        t_emb = self.time_embed(t)
        # Step 2: Apply self.input_conv to x
        x = self.input_conv(x)
        # Step 3: Encoder - for each (encoder_block, downsampler):
        #           - pass x through both ConvBlocks (with t_emb)
        #           - save x to a skip_connections list
        #           - downsample x
        skip_connections = []
        for enc_block, down_block in zip(self.encoder_blocks, self.downsamplers):
            x = enc_block[0](x, t_emb)
            x = enc_block[1](x, t_emb)
            skip_connections.append(x)
            x = down_block(x)
        # Step 4: Bottleneck - pass x through both bottleneck ConvBlocks
        x = self.bottleneck[0](x, t_emb)
        x = self.bottleneck[1](x, t_emb)
        # Step 5: Decoder - for each (upsampler, decoder_block):
        #           - upsample x
        #           - concatenate with skip_connections.pop() along dim=1
        #           - pass through both ConvBlocks (with t_emb)
        for dec_block, up_block in zip(self.decoder_blocks, self.upsamplers):
            x = up_block(x)
            x = torch.concat((x, skip_connections.pop()), dim=1)
            x = dec_block[0](x, t_emb)
            x = dec_block[1](x, t_emb)
        # Step 6: Apply self.output_conv to get final prediction
        x = self.output_conv(x)
        return x


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization for timestep conditioning.
    Normalizes features, then applies timestep-dependent scale and shift.
    """

    def __init__(self, num_channels, time_emb_dim, num_groups=32):
        """
        Args:
            num_channels: Number of feature channels to normalize (C)
            time_emb_dim: Dimension of timestep embedding input
            num_groups: Number of groups for GroupNorm (default 32)
        """
        super().__init__()

        # Standard group normalization (learnable affine disabled - we'll use our own scale/shift)
        self.group_norm = nn.GroupNorm(num_groups, num_channels, affine=False)

        # Project timestep embedding to scale and shift parameters
        # Input: (B, time_emb_dim) -> Output: (B, num_channels * 2) for scale AND shift
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, num_channels * 2)
        )

    def forward(self, x, t_emb):
        """
        Args:
            x: Feature tensor of shape (B, C, H, W)
            t_emb: Timestep embedding of shape (B, time_emb_dim)
        Returns:
            Modulated features of shape (B, C, H, W)
        """
        # Step 1: Apply group normalization to x
        x_norm = self.group_norm(x)
        # Step 2: Project t_emb through self.time_mlp to get scale and shift
        t_emb = self.time_mlp(t_emb)
        # Step 3: Split the projection into scale and shift (hint: chunk)
        scale, shift = torch.chunk(t_emb, 2, dim=1)
        # expand to apply broadcasting
        scale = scale[:,:, None, None]
        shift = shift[:,:, None, None]
        # Step 4: Apply modulation: scale * normalized + shift
        # the +1 is to account for the random init that is ~0 in the beginnig
        # nn.Linear uses Kaiming uniform init -> weights ~ Uniform(-1/√k, +1/√k) = Uniform(-0.088, +0.088)
        x_norm = (1+scale) * x_norm + shift 

        return x_norm
  

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=32):
        super().__init__()

        self.adaGN1 = AdaGN(in_channels, time_emb_dim, num_groups)
        self.conv_block1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        self.adaGN2 = AdaGN(out_channels, time_emb_dim, num_groups)
        self.conv_block2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )

        self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        res_connection = x
        x = self.adaGN1(x, t_emb)
        x = self.conv_block1(x)
        x = self.adaGN2(x, t_emb)
        x = self.conv_block2(x)
        x = x + self.skip(res_connection)
        return x


class Downsample(nn.Module):
    """Halves spatial dimensions using strided convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Doubles spatial dimensions using interpolation + convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encodes timestep integers into sinusoidal embeddings for diffusion models.
    Maps timesteps to high-dimensional vectors using sin/cos at multiple frequencies.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # Embedding dimension (e.g., 128 or 256)

    def forward(self, time):
        """
        Args:
            time: Tensor of shape (batch_size,) containing timestep integers
        Returns:
            embeddings: Tensor of shape (batch_size, dim) with sinusoidal encodings
        """
        device = time.device
        half_dim = self.dim // 2 # divice by to accomodate sin/cos positional embeddings

        # Create frequency bands: higher indices = higher frequencies
        # Formula: 10000^(-2i/dim) creates exponentially decreasing periods
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1) 
        # e.g., dim=128, half_dim=64, embeddings[0] = 1, slow frequency, big period; embeddings[63]= 0.0001, fast oscillation, tiny period
        # great importance: low frequency distinguish better timesteps that are far from each other e.g., 0 vs 500 --> sin(10*1.0) = -0.544,  sin(500*1.0) = -0.306  # Δ ≈ 0.238 (very different!)
        # high frequency instead distinguish better frequencies that are close to each other e.g., 100 vs 101 --> sin(100*0.0001) = 0.0099998,  sin(101*0.0001) = 0.0100998  # Δ ≈ 0.0001 (still distinguishable)

        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) 

        # Scale timesteps by frequency bands: (batch_size, 1) * (half_dim,) -> (batch_size, half_dim)
        embeddings = time[:, None] * embeddings[None, :]

        # Apply sin and cos, then concatenate: (batch_size, half_dim * 2)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings



# test adaGN
batch_sample = torch.randn((2, 3, 64, 64)).to('cuda')
t = (torch.ones((2))*36).to('cuda')
u_net = UNet(in_channels=3, out_channels=3).to('cuda')
pred = u_net(batch_sample, t)
print(pred.shape)
# test sinusoidal embeddings
# embedder = SinusoidalPositionEmbeddings(dim=16)
# t_early = embedder(torch.tensor([10]))    # Early denoising
# t_mid = embedder(torch.tensor([500]))     # Mid denoising
# t_mid2 = embedder(torch.tensor([501]))     # Mid denoising
# t_late = embedder(torch.tensor([990]))    # Late denoising

# print("Early timestep (10):", t_early[0, :4])  # First 4 dims
# print("Mid timestep (500):", t_mid[0, :4])
# print("Mid timestep 2 (501):", t_mid2[0, :4])
# print("Late timestep (990):", t_late[0, :4])