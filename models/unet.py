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


class ConvBlock(nn.Module):
    pass


class SinusoidalPositionEmbeddings(nn.Module):
    pass