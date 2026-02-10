import torch
import torch.nn as nn

class UNet(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.timestep_embedding = ...
        self.encoder = ...
        self.bottleneck = ...
        self.decoder = ...
        self.conditioning_mechanism = ...


    def forward(self):
        pass


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
sample = torch.randn((16, 64, 128, 128))
t_emb = torch.randn((16, 128))
res_net = ConvBlock(in_channels=64, out_channels=128, time_emb_dim=128, num_groups=32)
embs = res_net(sample, t_emb)


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