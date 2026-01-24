from setuptools import setup, find_packages

setup(
    name="video-diffusion-sr",
    version="0.1.0",
    description="Video Super-Resolution with Latent Diffusion Models",
    author="Your Name",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "einops>=0.6.1",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "tensorboard>=2.13.0",
        ],
    },
)
