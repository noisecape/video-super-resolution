# training/train.py
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from PIL import Image

from models.vae import VAE
from models.unet import UNet
from models.diffusion import NoiseScheduler


class Trainer:

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['device'])

        # Models
        self.vae = VAE().to(self.device)
        self.unet = UNet(in_channels=8, out_channels=4).to(self.device)
        self.scheduler = NoiseScheduler(num_timestep=config['num_timesteps'])

        # Move scheduler tensors to device (not nn.Module buffers)
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        # Optimizer (UNet only — VAE has no grad)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=config['lr'])
        self.scaler = GradScaler(enabled=(config['device'] == 'cuda'))

    def train_step(self, target_lr: torch.Tensor, target_hr: torch.Tensor) -> float:
        target_lr = target_lr.to(self.device)
        target_hr = target_hr.to(self.device)
        B = target_hr.shape[0]

        with torch.no_grad():
            # Encode HR to latent
            mean_hr, log_var_hr = self.vae.encode(target_hr)
            z_hr = self.vae.sample(mean_hr, log_var_hr)

            # Encode LR to latent at native size, then upsample to match HR latent
            mean_lr, log_var_lr = self.vae.encode(target_lr)
            z_lr_small = self.vae.sample(mean_lr, log_var_lr)
            z_lr = F.interpolate(z_lr_small, size=z_hr.shape[-2:], mode='bilinear', align_corners=False)

        t = torch.randint(0, self.config['num_timesteps'], (B,), device=self.device)
        x_t, noise = self.scheduler.add_noise(z_hr, t)

        # Concatenate noisy HR latent with LR conditioning along channel dim
        x_input = torch.cat([x_t, z_lr], dim=1)  # (B, 8, H, W)

        self.optimizer.zero_grad()
        with autocast(enabled=(self.config['device'] == 'cuda')):
            noise_pred = self.unet(x_input, t)
            loss = F.mse_loss(noise_pred, noise)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def train_epoch(self, dataloader) -> float:
        self.unet.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            if len(batch) == 3:
                _, target_lr, target_hr = batch   # (context_lr, target_lr, target_hr)
            else:
                target_lr, target_hr = batch      # (target_lr, target_hr)
            loss = self.train_step(target_lr, target_hr)
            total_loss += loss
            if (step + 1) % self.config['log_interval'] == 0:
                print(f"  step {step + 1}: loss={loss:.4f}")
        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader) -> float:
        self.unet.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    _, target_lr, target_hr = batch
                else:
                    target_lr, target_hr = batch
                target_lr = target_lr.to(self.device)
                target_hr = target_hr.to(self.device)
                B = target_hr.shape[0]

                mean_hr, log_var_hr = self.vae.encode(target_hr)
                z_hr = self.vae.sample(mean_hr, log_var_hr)

                mean_lr, log_var_lr = self.vae.encode(target_lr)
                z_lr_small = self.vae.sample(mean_lr, log_var_lr)
                z_lr = F.interpolate(z_lr_small, size=z_hr.shape[-2:], mode='bilinear', align_corners=False)

                t = torch.randint(0, self.config['num_timesteps'], (B,), device=self.device)
                x_t, noise = self.scheduler.add_noise(z_hr, t)
                x_input = torch.cat([x_t, z_lr], dim=1)
                noise_pred = self.unet(x_input, t)
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
        self.unet.train()
        return total_loss / len(dataloader)

    def validate_metrics(self, dataloader, num_samples: int, output_dir: str = None) -> dict:
        from training.metrics import psnr, ssim, _to_numpy_uint8_hwc
        self.unet.eval()
        psnr_scores, ssim_scores = [], []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_samples:
                    break
                if len(batch) == 3:
                    _, target_lr, target_hr = batch
                else:
                    target_lr, target_hr = batch
                target_lr = target_lr.to(self.device)
                target_hr = target_hr.to(self.device)

                # Encode LR once — fixed conditioning for the entire reverse loop
                mean_lr, log_var_lr = self.vae.encode(target_lr)
                z_lr_small = self.vae.sample(mean_lr, log_var_lr)

                # Get HR latent shape via a quick encode, then start from pure noise
                mean_hr, log_var_hr = self.vae.encode(target_hr)
                z_hr = self.vae.sample(mean_hr, log_var_hr)
                z_lr = F.interpolate(z_lr_small, size=z_hr.shape[-2:], mode='bilinear', align_corners=False)
                x_t = torch.randn_like(z_hr)

                # Reverse diffusion: T-1 → 0, conditioning on z_lr at every step
                for t in reversed(range(self.config['num_timesteps'])):
                    t_batch = torch.tensor([t], device=self.device)
                    x_input = torch.cat([x_t, z_lr], dim=1)
                    noise_pred = self.unet(x_input, t_batch)
                    x_t = self.scheduler.step(x_t, noise_pred, timestep=t)

                predicted = self.vae.decode(x_t).clamp(-1, 1)

                psnr_scores.append(psnr(predicted[0], target_hr[0]))
                ssim_scores.append(ssim(predicted[0], target_hr[0]))

                if output_dir:
                    Image.fromarray(_to_numpy_uint8_hwc(predicted[0])).save(
                        os.path.join(output_dir, f"sample_{i:02d}_pred.png")
                    )
                    Image.fromarray(_to_numpy_uint8_hwc(target_hr[0])).save(
                        os.path.join(output_dir, f"sample_{i:02d}_gt.png")
                    )

        self.unet.train()
        return {
            'psnr': sum(psnr_scores) / len(psnr_scores),
            'ssim': sum(ssim_scores) / len(ssim_scores),
        }

    def save_checkpoint(self, path: str, epoch: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(ckpt['unet_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        return ckpt['epoch']

    def train(self, train_loader, val_loader):
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        prev_ckpt_path = None
        for epoch in range(1, self.config['num_epochs'] + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            print(f"Epoch {epoch}/{self.config['num_epochs']} — train: {train_loss:.4f}, val: {val_loss:.4f}")

            if epoch % self.config['val_interval'] == 0:
                val_img_dir = os.path.join(self.config['checkpoint_dir'], f"val_images/epoch{epoch:04d}")
                metrics = self.validate_metrics(val_loader, self.config['val_num_samples'], output_dir=val_img_dir)
                print(f"  PSNR: {metrics['psnr']:.2f} dB  SSIM: {metrics['ssim']:.4f}")
                print(f"  Saved validation images: {val_img_dir}")

            ckpt_path = os.path.join(
                self.config['checkpoint_dir'], f"checkpoint_epoch{epoch:04d}.pt"
            )
            self.save_checkpoint(ckpt_path, epoch=epoch)
            print(f"Saved checkpoint: {ckpt_path}")

            if prev_ckpt_path and os.path.exists(prev_ckpt_path):
                os.remove(prev_ckpt_path)
            prev_ckpt_path = ckpt_path


def main():
    import torch.utils.data as data
    from data.vimeo90k import Vimeo90k

    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-4,
        'num_epochs': 100,
        'log_interval': 100,
        'checkpoint_dir': 'checkpoints/',
        'num_timesteps': 1000,
        'val_interval': 5,
        'val_num_samples': 8,
    }

    train_dataset = Vimeo90k(dataset_mode='train')
    train_loader = data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True
    )

    val_dataset = Vimeo90k(dataset_mode='test')
    val_loader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
