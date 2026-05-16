"""Shared test fixtures — synthetic data only, no dependence on trained models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.screening import ClassProfiles


@pytest.fixture(autouse=True)
def _seed_random():
    np.random.seed(0)
    yield


@pytest.fixture
def red_image() -> np.ndarray:
    """Solid red 64x64 RGB image — predictable HSV histogram."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 220
    img[..., 1] = 30
    img[..., 2] = 30
    return img


@pytest.fixture
def blue_image() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 30
    img[..., 1] = 30
    img[..., 2] = 220
    return img


@pytest.fixture
def red_image_file(tmp_path: Path, red_image) -> Path:
    p = tmp_path / "red.jpg"
    Image.fromarray(red_image).save(p)
    return p


@pytest.fixture
def synthetic_profiles() -> ClassProfiles:
    """3-class profile with distance statistics — small enough to test fast."""
    classes = ["red_flower", "blue_flower", "green_flower"]
    n = len(classes)

    color_dim = 1024
    texture_dim = 26
    rng = np.random.RandomState(0)
    color_means = rng.rand(n, color_dim).astype(np.float32)
    color_means /= color_means.sum(axis=1, keepdims=True)
    texture_means = rng.rand(n, texture_dim).astype(np.float32)
    texture_means /= texture_means.sum(axis=1, keepdims=True)

    color_dist_means = np.full(n, 5.0, dtype=np.float32)
    color_dist_stds = np.full(n, 0.5, dtype=np.float32)
    texture_dist_means = np.full(n, 1.0, dtype=np.float32)
    texture_dist_stds = np.full(n, 0.1, dtype=np.float32)

    return ClassProfiles(
        classes=classes,
        color_means=color_means,
        texture_means=texture_means,
        color_dist_means=color_dist_means,
        color_dist_stds=color_dist_stds,
        texture_dist_means=texture_dist_means,
        texture_dist_stds=texture_dist_stds,
    )
