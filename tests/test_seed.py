from __future__ import annotations

import random

import numpy as np

from src.seed import DEFAULT_SEED, set_global_seed


def test_python_random_is_deterministic():
    set_global_seed(42)
    a = [random.random() for _ in range(5)]
    set_global_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_is_deterministic():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_diverge():
    set_global_seed(1)
    a = np.random.rand(5)
    set_global_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_default_seed_constant():
    assert DEFAULT_SEED == 42


def test_returns_seed():
    assert set_global_seed(7) == 7
