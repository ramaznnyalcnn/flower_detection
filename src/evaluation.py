from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_utils import write_json


def _predict_proba(estimator, x):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return None


def top_k_accuracy(y_true, probabilities, classes, k: int = 5) -> float | None:
    if probabilities is None:
        return None
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    k = min(k, probabilities.shape[1])
    top_indices = np.argsort(probabilities, axis=1)[:, -k:]
    top_labels = classes[top_indices]
    return float(np.mean([label in row for label, row in zip(y_true, top_labels)]))


def save_confusion_matrix_png(
    y_true,
    y_pred,
    classes: list[str],
    output_path: str | Path,
    normalize: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    size = max(8, min(28, len(classes) * 0.22))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_step = max(1, len(classes) // 30)
    tick_positions = np.arange(0, len(classes), tick_step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([classes[i] for i in tick_positions], rotation=90, fontsize=6)
    ax.set_yticklabels([classes[i] for i in tick_positions], fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_predictions(
    y_true,
    y_pred,
    probabilities=None,
    classes: list[str] | None = None,
    output_dir: str | Path | None = None,
    prefix: str = "model",
    elapsed_seconds: float | None = None,
) -> dict:
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": top_k_accuracy(y_true, probabilities, classes, k=5),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples": int(len(y_true)),
    }
    if elapsed_seconds is not None:
        metrics["elapsed_seconds"] = float(elapsed_seconds)
        metrics["inference_ms_per_image"] = float((elapsed_seconds / max(len(y_true), 1)) * 1000)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
        write_json(output_dir / f"{prefix}_metrics.json", metrics)
        write_json(output_dir / f"{prefix}_classification_report.json", report)
        save_confusion_matrix_png(y_true, y_pred, classes, output_dir / f"{prefix}_confusion_matrix.png")

    return metrics


def evaluate_estimator(estimator, x, y, classes: list[str], output_dir: str | Path, prefix: str) -> dict:
    start = time.perf_counter()
    y_pred = estimator.predict(x)
    elapsed = time.perf_counter() - start
    probabilities = _predict_proba(estimator, x)
    estimator_classes = list(getattr(estimator, "classes_", classes))
    return evaluate_predictions(
        y,
        y_pred,
        probabilities=probabilities,
        classes=estimator_classes,
        output_dir=output_dir,
        prefix=prefix,
        elapsed_seconds=elapsed,
    )


def write_metrics_summary(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    keys = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "model",
        "split",
        "accuracy",
        "top5_accuracy",
        "f1_macro",
        "f1_weighted",
        "samples",
        "elapsed_seconds",
        "inference_ms_per_image",
    ]
    fieldnames = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: str | Path, title: str, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Metrics", ""]
    if rows:
        columns = ["model", "split", "accuracy", "top5_accuracy", "f1_macro", "samples"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("No metrics written yet.")
    lines.extend(["", "## Notes", "", "- Generated by run.py."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

