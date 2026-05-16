from __future__ import annotations

import numpy as np
import pytest

from src.screening import ClassProfiles, _bhattacharyya, _chi2


def test_bhattacharyya_identical_is_zero():
    p = np.array([0.5, 0.5])
    assert _bhattacharyya(p, p) == pytest.approx(0.0, abs=1e-6)


def test_bhattacharyya_disjoint_is_large():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert _bhattacharyya(p, q) > 5.0


def test_chi2_identical_is_zero():
    p = np.array([0.5, 0.5])
    assert _chi2(p, p) == pytest.approx(0.0, abs=1e-9)


def test_chi2_symmetric():
    p = np.array([0.2, 0.8])
    q = np.array([0.7, 0.3])
    assert _chi2(p, q) == pytest.approx(_chi2(q, p), abs=1e-9)


def test_save_load_roundtrip(tmp_path, synthetic_profiles):
    path = tmp_path / "p.npz"
    synthetic_profiles.save(path)
    loaded = ClassProfiles.load(path)

    assert loaded.classes == synthetic_profiles.classes
    np.testing.assert_allclose(loaded.color_means, synthetic_profiles.color_means)
    np.testing.assert_allclose(loaded.texture_means, synthetic_profiles.texture_means)
    np.testing.assert_allclose(loaded.color_dist_means, synthetic_profiles.color_dist_means)
    np.testing.assert_allclose(loaded.color_dist_stds, synthetic_profiles.color_dist_stds)


def test_save_load_backward_compat(tmp_path):
    """Old profiles without distance stats must still load."""
    path = tmp_path / "old.npz"
    classes = ["a", "b"]
    color_means = np.eye(2, 1024, dtype=np.float32)
    texture_means = np.eye(2, 26, dtype=np.float32)
    np.savez(path, classes=np.array(classes), color_means=color_means, texture_means=texture_means)

    loaded = ClassProfiles.load(path)
    assert loaded.classes == classes
    assert loaded.color_dist_means is None
    assert loaded.color_dist_stds is None


def test_adaptive_verdict_definitely_not(synthetic_profiles):
    """color_percentile > 90 with stats present → reject."""
    verdict = synthetic_profiles.get_adaptive_verdict(
        "red_flower",
        color_distance=100.0,
        texture_distance=0.0,
        color_percentile=95.0,
        texture_percentile=0.0,
    )
    assert verdict == "kesinlikle bu olamaz"


def test_adaptive_verdict_probably_not(synthetic_profiles):
    """color_distance above mean+2*std → probably not."""
    verdict = synthetic_profiles.get_adaptive_verdict(
        "red_flower",
        color_distance=10.0,
        texture_distance=0.0,
        color_percentile=50.0,
        texture_percentile=0.0,
    )
    assert verdict == "muhtemelen değil"


def test_adaptive_verdict_could_be(synthetic_profiles):
    verdict = synthetic_profiles.get_adaptive_verdict(
        "red_flower",
        color_distance=4.5,
        texture_distance=0.0,
        color_percentile=10.0,
        texture_percentile=0.0,
    )
    assert verdict == "olabilir"


def test_adaptive_verdict_unknown_label(synthetic_profiles):
    verdict = synthetic_profiles.get_adaptive_verdict(
        "nonexistent",
        color_distance=1.0,
        texture_distance=0.0,
    )
    assert verdict == "kesinlikle bu olamaz"


def test_adaptive_verdict_fallback_no_stats():
    """Without distance stats, falls back to percentile thresholds."""
    profiles = ClassProfiles(
        classes=["a", "b"],
        color_means=np.eye(2, 1024, dtype=np.float32),
        texture_means=np.eye(2, 26, dtype=np.float32),
    )
    assert profiles.get_adaptive_verdict("a", 1.0, 1.0, color_percentile=95.0) == "kesinlikle bu olamaz"
    assert profiles.get_adaptive_verdict("a", 1.0, 1.0, color_percentile=85.0) == "muhtemelen değil"
    assert profiles.get_adaptive_verdict("a", 1.0, 1.0, color_percentile=10.0) == "olabilir"


def test_rank_color_returns_all_classes(synthetic_profiles, red_image):
    ranks = synthetic_profiles.rank_color(red_image)
    assert len(ranks) == len(synthetic_profiles.classes)
    assert {cls for cls, _, _ in ranks} == set(synthetic_profiles.classes)
    # Sorted ascending by distance
    dists = [d for _, d, _ in ranks]
    assert dists == sorted(dists)


def test_rank_texture_percentiles_in_range(synthetic_profiles, red_image):
    ranks = synthetic_profiles.rank_texture(red_image)
    for _, _, pct in ranks:
        assert 0.0 <= pct <= 100.0
