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

## Project Structure

```
├── data/                 # Dataset loading and preprocessing
├── models/              # Model architectures
├── training/            # Training pipeline
├── configs/             # Configuration files
├── scripts/             # Execution scripts
├── notebooks/           # Jupyter notebooks for exploration
└── demos/outputs/       # Demo outputs
```

## Hardware I used

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