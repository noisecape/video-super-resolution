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


def test_encoder_output_shape():
    encoder = Encoder()
    x = torch.randn(2, 3, 64, 64)
    out = encoder(x)
    assert out.shape == (2, 8, 8, 8)  # 8x compression, 8ch = mean(4) + log_var(4)

def test_encoder_output_channels():
    encoder = Encoder()
    assert encoder.output_conv[-1].out_channels == 8


def test_decoder_output_shape():
    decoder = Decoder()
    z = torch.randn(2, 4, 8, 8)
    out = decoder(z)
    assert out.shape == (2, 3, 64, 64)

def test_decoder_output_channels():
    decoder = Decoder()
    assert decoder.output_conv[-1].out_channels == 3