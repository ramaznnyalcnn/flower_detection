"""SQLite-backed prediction log for offline analysis.

Each call to predict_pipeline can drop one row here; downstream
scripts (analyze_predictions.py, run_eval.py) read the same table
to produce per-class accuracy, abstain rates, latency histograms.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

DEFAULT_DB_PATH = Path("logs/predictions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    image_path TEXT,
    image_hash TEXT,
    ground_truth TEXT,
    source TEXT NOT NULL,
    model_path TEXT,
    top1_label TEXT,
    top1_score REAL,
    top1_fusion REAL,
    top1_abstain INTEGER,
    top2_label TEXT,
    top2_score REAL,
    top2_fusion REAL,
    top3_label TEXT,
    top3_score REAL,
    top3_fusion REAL,
    bg_removed INTEGER,
    latency_ms REAL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
CREATE INDEX IF NOT EXISTS idx_predictions_source ON predictions(source);
CREATE INDEX IF NOT EXISTS idx_predictions_top1 ON predictions(top1_label);
"""


def _hash_file(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(
    image_path: str | Path,
    results: list,  # list[ScreeningResult]
    *,
    source: str,
    model_path: str | Path,
    latency_ms: float,
    ground_truth: str | None = None,
    notes: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Persist a single prediction row. Returns the inserted row id.
    Silently no-ops on I/O errors (logging must never block inference)."""
    try:
        image_path = Path(image_path)
        image_hash = _hash_file(image_path) if image_path.exists() else None

        def at(n: int, attr: str, default=None):
            return getattr(results[n], attr, default) if n < len(results) else default

        with _connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO predictions (
                    ts, image_path, image_hash, ground_truth, source, model_path,
                    top1_label, top1_score, top1_fusion, top1_abstain,
                    top2_label, top2_score, top2_fusion,
                    top3_label, top3_score, top3_fusion,
                    bg_removed, latency_ms, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    str(image_path),
                    image_hash,
                    ground_truth,
                    source,
                    str(model_path),
                    at(0, "label"), at(0, "model_score"), at(0, "fusion_score"),
                    int(bool(at(0, "abstain", False))),
                    at(1, "label"), at(1, "model_score"), at(1, "fusion_score"),
                    at(2, "label"), at(2, "model_score"), at(2, "fusion_score"),
                    int(bool(at(0, "bg_removed", True))),
                    latency_ms,
                    notes,
                ),
            )
            return cur.lastrowid
    except Exception as exc:
        print(f"[uyarı] prediction log atlandı: {exc}")
        return -1


def read_predictions(db_path: Path = DEFAULT_DB_PATH, source: str | None = None) -> list[dict]:
    if not db_path.exists():
        return []
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if source:
            cur = conn.execute("SELECT * FROM predictions WHERE source = ? ORDER BY ts", (source,))
        else:
            cur = conn.execute("SELECT * FROM predictions ORDER BY ts")
        return [dict(row) for row in cur.fetchall()]
