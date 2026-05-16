from __future__ import annotations

import numpy as np

from src.background_removal import fg_ratio, is_likely_flower, remove_background


def test_fg_ratio_empty():
    mask = np.zeros((20, 20), dtype=np.uint8)
    assert fg_ratio(mask) == 0.0


def test_fg_ratio_full():
    mask = np.ones((20, 20), dtype=np.uint8)
    assert fg_ratio(mask) == 1.0


def test_fg_ratio_half():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, :10] = 1
    assert fg_ratio(mask) == 0.5


def test_is_likely_flower_too_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 1  # 1% foreground
    assert is_likely_flower(mask) is False


def test_is_likely_flower_too_full():
    mask = np.ones((10, 10), dtype=np.uint8)
    assert is_likely_flower(mask) is False


def test_is_likely_flower_good_range():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :] = 1  # 50% foreground
    assert is_likely_flower(mask) is True


def test_is_likely_flower_custom_bounds():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, :] = 1  # 10% foreground
    assert is_likely_flower(mask, min_ratio=0.05, max_ratio=0.5) is True
    assert is_likely_flower(mask, min_ratio=0.20, max_ratio=0.5) is False


def test_remove_background_shape(red_image_file):
    masked, mask = remove_background(red_image_file, image_size=64)
    assert masked.shape == (64, 64, 3)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
