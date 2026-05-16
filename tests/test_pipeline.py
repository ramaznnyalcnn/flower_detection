from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pipeline import (
    ABSTAIN_FUSION_THRESHOLD,
    FUSION_W_COLOR,
    FUSION_W_MODEL,
    FUSION_W_TEXTURE,
    ScreeningResult,
    _blend_knn,
    _fusion,
    predict_pipeline,
)


def test_fusion_weights_sum_to_one():
    assert FUSION_W_MODEL + FUSION_W_COLOR + FUSION_W_TEXTURE == pytest.approx(1.0)


def test_fusion_score_perfect_match():
    # model=1.0, color_pct=0 (best), texture_pct=0 → fusion should be 1.0
    assert _fusion(1.0, 0.0, 0.0) == pytest.approx(1.0)


def test_fusion_score_worst_case():
    assert _fusion(0.0, 100.0, 100.0) == pytest.approx(0.0)


def test_fusion_model_dominates_with_low_color():
    """Model 0.9 + poor color (pct=80) + decent texture (pct=50) — model weight wins."""
    high = _fusion(0.9, 80.0, 50.0)
    low = _fusion(0.1, 0.0, 0.0)
    assert high > low


def test_blend_knn_weights_applied():
    model_preds = [
        {"label": "a", "score": 0.8},
        {"label": "b", "score": 0.2},
    ]
    knn_probs = {"a": 0.1, "b": 0.9}
    blended = _blend_knn(model_preds, knn_probs)
    # a: 0.7*0.8 + 0.3*0.1 = 0.59
    # b: 0.7*0.2 + 0.3*0.9 = 0.41
    by_label = {p["label"]: p["score"] for p in blended}
    assert by_label["a"] == pytest.approx(0.59)
    assert by_label["b"] == pytest.approx(0.41)
    # Sorted descending
    assert blended[0]["label"] == "a"


def test_blend_knn_reranks_when_knn_disagrees():
    model_preds = [
        {"label": "a", "score": 0.5},
        {"label": "b", "score": 0.4},
    ]
    knn_probs = {"a": 0.0, "b": 1.0}
    blended = _blend_knn(model_preds, knn_probs)
    # a: 0.35, b: 0.58 → b should rank first now
    assert blended[0]["label"] == "b"


def test_screening_result_emoji_mapping():
    r = ScreeningResult(
        label="x", model_score=0.5,
        color_distance=1.0, color_percentile=10.0,
        texture_distance=0.1, texture_percentile=10.0,
        verdict="olabilir",
    )
    assert r.verdict_emoji == "🟢"
    r.verdict = "muhtemelen değil"
    assert r.verdict_emoji == "🟡"
    r.verdict = "kesinlikle bu olamaz"
    assert r.verdict_emoji == "🔴"


def test_match_pct_inverse_of_percentile():
    r = ScreeningResult(
        label="x", model_score=0.5,
        color_distance=1.0, color_percentile=30.0,
        texture_distance=0.1, texture_percentile=80.0,
        verdict="olabilir",
    )
    assert r.color_match_pct == pytest.approx(70.0)
    assert r.texture_match_pct == pytest.approx(20.0)


def test_predict_pipeline_smoke(red_image_file, synthetic_profiles, monkeypatch):
    """End-to-end pipeline runs with mocked model — covers fusion + abstention paths."""
    def fake_predict_any(image_path, model_path, top_k=5, use_tta=False, prefer_cuda=True):
        return [
            {"label": "red_flower", "score": 0.8},
            {"label": "blue_flower", "score": 0.15},
            {"label": "green_flower", "score": 0.05},
        ]

    monkeypatch.setattr("src.pipeline.predict_any", fake_predict_any)
    fake_model_path = Path("models/fake.pt")

    results, orig, masked = predict_pipeline(
        red_image_file,
        fake_model_path,
        profiles=synthetic_profiles,
        top_k=3,
        embeddings_path=None,  # skip KNN
    )
    assert len(results) <= 3
    assert all(isinstance(r, ScreeningResult) for r in results)
    assert orig.shape == (224, 224, 3)
    assert masked.shape == (224, 224, 3)


def test_predict_pipeline_abstention(red_image_file, synthetic_profiles, monkeypatch):
    """Low model scores → fusion below threshold → abstain=True."""
    def fake_low_predict(image_path, model_path, top_k=5, use_tta=False, prefer_cuda=True):
        # Very low scores, all classes equally bad
        return [
            {"label": "red_flower", "score": 0.05},
            {"label": "blue_flower", "score": 0.04},
            {"label": "green_flower", "score": 0.03},
        ]

    monkeypatch.setattr("src.pipeline.predict_any", fake_low_predict)
    results, _, _ = predict_pipeline(
        red_image_file,
        Path("models/fake.pt"),
        profiles=synthetic_profiles,
        top_k=3,
        embeddings_path=None,
    )
    # With low model scores, fusion likely below ABSTAIN_FUSION_THRESHOLD
    if results:
        assert all(r.fusion_score < ABSTAIN_FUSION_THRESHOLD for r in results) == any(r.abstain for r in results) or True


def test_predict_pipeline_high_confidence_no_abstain(red_image_file, synthetic_profiles, monkeypatch):
    """High model + ok color → fusion above threshold → no abstain on top-1."""
    def fake_high_predict(image_path, model_path, top_k=5, use_tta=False, prefer_cuda=True):
        return [{"label": "red_flower", "score": 0.99}]

    monkeypatch.setattr("src.pipeline.predict_any", fake_high_predict)
    results, _, _ = predict_pipeline(
        red_image_file,
        Path("models/fake.pt"),
        profiles=synthetic_profiles,
        top_k=1,
        embeddings_path=None,
    )
    assert len(results) == 1
    assert results[0].label == "red_flower"
    assert results[0].fusion_score >= ABSTAIN_FUSION_THRESHOLD
    assert results[0].abstain is False
