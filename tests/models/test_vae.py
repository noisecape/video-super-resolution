# tests/models/test_vae.py
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


def test_vae_encode_output_shapes():
    vae = VAE()
    x = torch.randn(2, 3, 64, 64)
    mean, log_var = vae.encode(x)
    assert mean.shape == (2, 4, 8, 8)
    assert log_var.shape == (2, 4, 8, 8)

def test_vae_sample_output_shape():
    vae = VAE()
    mean = torch.zeros(2, 4, 8, 8)
    log_var = torch.zeros(2, 4, 8, 8)
    z = vae.sample(mean, log_var)
    assert z.shape == (2, 4, 8, 8)

def test_vae_sample_is_stochastic():
    vae = VAE()
    mean = torch.zeros(2, 4, 8, 8)
    log_var = torch.zeros(2, 4, 8, 8)
    z1 = vae.sample(mean, log_var)
    z2 = vae.sample(mean, log_var)
    assert not torch.allclose(z1, z2)

def test_vae_decode_output_shape():
    vae = VAE()
    z = torch.randn(2, 4, 8, 8)
    out = vae.decode(z)
    assert out.shape == (2, 3, 64, 64)

def test_vae_forward_output_shapes():
    vae = VAE()
    x = torch.randn(2, 3, 64, 64)
    recon, mean, log_var = vae.forward(x)
    assert recon.shape == (2, 3, 64, 64)
    assert mean.shape == (2, 4, 8, 8)
    assert log_var.shape == (2, 4, 8, 8)

def test_kl_loss_zero_for_unit_gaussian():
    vae = VAE()
    mean = torch.zeros(2, 4, 8, 8)
    log_var = torch.zeros(2, 4, 8, 8)
    loss = vae.kl_loss(mean, log_var)
    assert torch.isclose(loss, torch.tensor(0.0))

def test_kl_loss_non_negative():
    vae = VAE()
    mean = torch.randn(2, 4, 8, 8)
    log_var = torch.randn(2, 4, 8, 8)
    loss = vae.kl_loss(mean, log_var)
    assert loss >= 0.0