# tests/data/test_vimeo90k.py
import numpy as np
import pytest
import tempfile
import cv2
from pathlib import Path
from unittest import mock

from data.vimeo90k import Vimeo90k


def make_fake_dataset(tmpdir: str, num_sequences: int = 2):
    """Build a minimal Vimeo-90K directory tree with tiny synthetic images."""
    root = Path(tmpdir)
    sequences_dir = root / 'sequences'
    seq_ids = []
    for i in range(num_sequences):
        seq_path = sequences_dir / f'0000{i+1}' / '0001'
        seq_path.mkdir(parents=True)
        for f in range(1, 8):  # im1.png ... im7.png
            # Tiny 16x16 RGB image, values 100-200 so normalization is testable
            img = np.full((16, 16, 3), 150, dtype=np.uint8)
            cv2.imwrite(str(seq_path / f'im{f}.png'), img)
        seq_ids.append(f'0000{i+1}/0001')

    # Write sequence list files
    for name in ('sep_trainlist.txt', 'sep_testlist.txt'):
        (root / name).write_text('\n'.join(seq_ids) + '\n')

    return root


def make_dataset(tmpdir, **kwargs):
    root = make_fake_dataset(tmpdir, num_sequences=2)
    with mock.patch('data.vimeo90k._DATASET_ROOT', root):
        return Vimeo90k(img_resolution=(16, 16), **kwargs)


def test_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = make_dataset(tmpdir)
        assert len(ds) == 2


def test_getitem_returns_three_tensors():
    import torch
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = make_dataset(tmpdir)
        item = ds[0]
        assert len(item) == 3
        context_lr, target_lr, target_hr = item
        assert all(hasattr(t, 'shape') for t in (context_lr, target_lr, target_hr))


def test_getitem_shapes():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = make_dataset(tmpdir)
        context_lr, target_lr, target_hr = ds[0]
        # context_lr: 6 LR frames, each (C, H//factor, W//factor)
        assert context_lr.shape == (6, 3, 8, 8)   # downscale_factor=2 default
        # target_lr / target_hr: single frame
        assert target_lr.shape == (3, 8, 8)
        assert target_hr.shape == (3, 16, 16)


def test_getitem_dtype():
    import torch
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = make_dataset(tmpdir)
        context_lr, target_lr, target_hr = ds[0]
        assert context_lr.dtype == torch.float32
        assert target_lr.dtype == torch.float32
        assert target_hr.dtype == torch.float32


def test_getitem_value_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = make_dataset(tmpdir)
        context_lr, target_lr, target_hr = ds[0]
        for t in (context_lr, target_lr, target_hr):
            assert t.min() >= -1.0 - 1e-5
            assert t.max() <= 1.0 + 1e-5
