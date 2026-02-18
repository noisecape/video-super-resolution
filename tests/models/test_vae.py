# tests/models/test_vae.py
import pytest
import torch
import torch.nn as nn
from models.vae import ResBlock, Encoder, Decoder, VAE

def test_resblock_same_channels():
    block = ResBlock(128, 128)
    x = torch.randn(2, 128, 32, 32)
    out = block(x)
    assert out.shape == (2, 128, 32, 32)

def test_resblock_different_channels():
    block = ResBlock(128, 256)
    x = torch.randn(2, 128, 32, 32)
    out = block(x)
    assert out.shape == (2, 256, 32, 32)

def test_resblock_residual_skip():
    block = ResBlock(128, 256)
    assert not isinstance(block.skip, nn.Identity)

def test_resblock_identity_skip():
    block = ResBlock(128, 128)
    assert isinstance(block.skip, nn.Identity)
