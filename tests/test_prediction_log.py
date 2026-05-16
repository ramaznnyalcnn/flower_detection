from __future__ import annotations

from pathlib import Path

from src.pipeline import ScreeningResult
from src.prediction_log import log_prediction, read_predictions


def _make_result(label: str, score: float, fusion: float, abstain: bool = False) -> ScreeningResult:
    return ScreeningResult(
        label=label, model_score=score,
        color_distance=1.0, color_percentile=10.0,
        texture_distance=0.1, texture_percentile=10.0,
        verdict="olabilir", bg_removed=True,
        fusion_score=fusion, abstain=abstain,
    )


def test_log_and_read_roundtrip(tmp_path, red_image_file):
    db = tmp_path / "test.db"
    results = [
        _make_result("rose", 0.9, 0.85),
        _make_result("tulip", 0.05, 0.30, abstain=True),
    ]
    rowid = log_prediction(
        red_image_file, results,
        source="test", model_path="fake.pt", latency_ms=12.3,
        ground_truth="rose",
        db_path=db,
    )
    assert rowid > 0

    rows = read_predictions(db, source="test")
    assert len(rows) == 1
    row = rows[0]
    assert row["top1_label"] == "rose"
    assert row["top1_score"] == 0.9
    assert row["top1_fusion"] == 0.85
    assert row["top1_abstain"] == 0
    assert row["top2_label"] == "tulip"
    assert row["ground_truth"] == "rose"
    assert row["source"] == "test"
    assert row["latency_ms"] == 12.3
    assert row["image_hash"] is not None
    assert len(row["image_hash"]) == 64


def test_log_handles_empty_results(tmp_path, red_image_file):
    db = tmp_path / "test.db"
    rowid = log_prediction(
        red_image_file, [],
        source="test", model_path="fake.pt", latency_ms=1.0,
        db_path=db,
    )
    assert rowid > 0
    rows = read_predictions(db, source="test")
    assert len(rows) == 1
    assert rows[0]["top1_label"] is None


def test_read_predictions_returns_empty_for_missing_db(tmp_path):
    rows = read_predictions(tmp_path / "nonexistent.db")
    assert rows == []


def test_log_filters_by_source(tmp_path, red_image_file):
    db = tmp_path / "test.db"
    log_prediction(red_image_file, [_make_result("a", 0.5, 0.5)],
                   source="app", model_path="m.pt", latency_ms=1.0, db_path=db)
    log_prediction(red_image_file, [_make_result("b", 0.5, 0.5)],
                   source="eval", model_path="m.pt", latency_ms=1.0, db_path=db)

    assert len(read_predictions(db, source="app")) == 1
    assert len(read_predictions(db, source="eval")) == 1
    assert len(read_predictions(db)) == 2
