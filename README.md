# Video Super-Resolution with Diffusion Models

Latent diffusion model for video super-resolution (480p→720p) with temporal consistency mechanisms.

## Installation

### Create conda environment (recommended):

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate video-sr

# Install package in development mode
pip install -e .
```

### Or use pip only:

```bash
pip install -r requirements.txt
pip install -e .
```

This allows you to import modules across the project:
```python
from data.dataset import Vimeo90KDataset
from models.unet import UNet
from training.train import train
```

## Project Structure

```
├── data/                 # Dataset loading and preprocessing
│   ├── dataset.py       # Vimeo-90K dataloader
│   ├── preprocessing.py # Video processing utilities
│   └── augmentation.py  # Data augmentation
├── models/              # Model architectures
│   ├── unet.py         # 2D/3D U-Net architecture
│   ├── vae.py          # VAE encoder/decoder
│   ├── diffusion.py    # Diffusion process
│   └── temporal.py     # Temporal consistency modules
├── training/            # Training pipeline
│   ├── train.py        # Main training loop
│   ├── losses.py       # Loss functions
│   └── metrics.py      # Evaluation metrics
├── configs/             # Configuration files
├── scripts/             # Execution scripts
├── notebooks/           # Jupyter notebooks for exploration
└── demos/outputs/       # Demo outputs
```

## Hardware Requirements

- GPU: RTX 3090Ti (24GB VRAM) or equivalent
- Mixed precision training (fp16) required
- Dataset: ~90GB for Vimeo-90K

## Usage

```bash
# Training
python scripts/train.py --config configs/base_config.yaml

# Evaluation
python scripts/evaluate.py --checkpoint path/to/model.pth

# Inference
python scripts/inference.py --input video.mp4 --output upscaled.mp4
```

## Research Focus

Primary contribution: Novel temporal consistency mechanism for video diffusion

**Target Metrics:**
- PSNR > 30 dB
- SSIM > 0.85
- LPIPS < 0.15
- Temporal consistency (custom metric)
