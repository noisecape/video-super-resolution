import torch.nn as nn
import torch


class NoiseScheduler(nn.Module):

    def __init__(self, num_timestep:int=1000, beta_start:int=0.0001, beta_end:int=0.02):
        super().__init__()

        self.num_timestep = num_timestep

        # noise scheduler: e.g., starts with 0.0001 ends with 0.02
        self.betas = torch.linspace(beta_start, beta_end, num_timestep)

        self.alphas = 1.0 - self.betas
        # calculate comulative product, i.e., alpha_0 * alpha_1 * alpha_2 * ... * alpha_t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    
    def add_noise(self, x_0, timesteps):
        # corruption step formula: x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε
        noise = torch.randn_like(x_0)
        # reshape to (B, 1, 1, 1) so scalars broadcast correctly over (B, C, H, W)
        ndim = x_0.ndim
        shape = (-1,) + (1,) * (ndim - 1)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[timesteps]).view(shape)
        sqrt_one_minus = torch.sqrt(1 - self.alphas_cumprod[timesteps]).view(shape)
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus * noise
        return x_t, noise
    

    def step(self, x_t, model_output, timestep):
        # reverse progress x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_pred) + σ_t * z
        alpha_t = self.alphas[timestep]
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        beta_t = self.betas[timestep]

        # compute predicted x{t-1} mean
        # μ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_pred)
        pred_mean = (1/torch.sqrt(alpha_t)) * (x_t - (beta_t/torch.sqrt(1-alpha_cumprod_t)) * model_output)

        ## Add noise (excluding final step)
        if timestep > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_prev = pred_mean + (sigma_t * noise)
        else:
            x_prev = pred_mean
        
        return x_prev