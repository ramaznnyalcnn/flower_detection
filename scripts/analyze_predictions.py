"""Read logs/predictions.db, produce per-class accuracy + abstain + latency.

Writes:
    results/analysis/test_report.md
    results/analysis/per_class.csv
    results/analysis/top_confusions.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction_log import DEFAULT_DB_PATH, read_predictions


def analyze(rows: list[dict]) -> dict:
    if not rows:
        return {"total": 0}

    total = len(rows)
    with_gt = [r for r in rows if r.get("ground_truth")]
    abstain_count = sum(1 for r in rows if r["top1_abstain"])
    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    fusion_scores = [r["top1_fusion"] for r in rows if r["top1_fusion"] is not None]

    per_class = defaultdict(lambda: {"n": 0, "correct": 0, "abstain": 0})
    confusions = Counter()
    for r in with_gt:
        gt = r["ground_truth"]
        pred = r["top1_label"]
        per_class[gt]["n"] += 1
        if r["top1_abstain"]:
            per_class[gt]["abstain"] += 1
        if pred == gt:
            per_class[gt]["correct"] += 1
        elif pred:
            confusions[(gt, pred)] += 1

    per_class_rows = []
    for cls, stats in sorted(per_class.items()):
        n = stats["n"]
        per_class_rows.append({
            "class": cls,
            "n": n,
            "accuracy": stats["correct"] / n if n else 0.0,
            "abstain_rate": stats["abstain"] / n if n else 0.0,
        })

    overall_acc = (
        sum(s["correct"] for s in per_class.values())
        / sum(s["n"] for s in per_class.values())
        if per_class else 0.0
    )

    return {
        "total": total,
        "with_gt": len(with_gt),
        "overall_accuracy": overall_acc,
        "abstain_rate": abstain_count / total,
        "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p95_ms": (sorted(latencies)[int(0.95 * len(latencies))]
                          if len(latencies) >= 20 else max(latencies, default=0.0)),
        "fusion_mean": sum(fusion_scores) / len(fusion_scores) if fusion_scores else 0.0,
        "per_class": per_class_rows,
        "top_confusions": confusions.most_common(20),
    }


def write_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    md = out_dir / "test_report.md"
    with md.open("w") as f:
        f.write("# Tahmin Analiz Raporu\n\n")
        f.write(f"- Toplam tahmin: **{report['total']}**\n")
        f.write(f"- Ground truth'lu: **{report['with_gt']}**\n")
        if report["with_gt"]:
            f.write(f"- Genel doğruluk (top-1): **{report['overall_accuracy']:.4f}**\n")
        f.write(f"- Abstain oranı: **{report['abstain_rate']:.4f}**\n")
        f.write(f"- Ortalama latency: **{report['latency_mean_ms']:.1f} ms**\n")
        f.write(f"- p95 latency: **{report['latency_p95_ms']:.1f} ms**\n")
        f.write(f"- Ortalama fusion skoru: **{report['fusion_mean']:.3f}**\n\n")

        if report.get("per_class"):
            f.write("## Sınıf Başına Performans (en düşük 20)\n\n")
            f.write("| Sınıf | N | Acc | Abstain |\n|---|---|---|---|\n")
            worst = sorted(report["per_class"], key=lambda r: r["accuracy"])[:20]
            for row in worst:
                f.write(f"| {row['class']} | {row['n']} | {row['accuracy']:.3f} | {row['abstain_rate']:.3f} |\n")
            f.write("\n")

        if report.get("top_confusions"):
            f.write("## En Sık Karışıklıklar\n\n")
            f.write("| Gerçek | Tahmin | Sayı |\n|---|---|---|\n")
            for (gt, pred), n in report["top_confusions"]:
                f.write(f"| {gt} | {pred} | {n} |\n")

    if report.get("per_class"):
        with (out_dir / "per_class.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class", "n", "accuracy", "abstain_rate"])
            w.writeheader()
            for row in report["per_class"]:
                w.writerow(row)

    if report.get("top_confusions"):
        with (out_dir / "top_confusions.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ground_truth", "predicted", "count"])
            for (gt, pred), n in report["top_confusions"]:
                w.writerow([gt, pred, n])

    print(f"Wrote {md}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--source", default="eval",
                        help="Filter by source (eval/app/pipeline). Empty = all.")
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args()

    rows = read_predictions(Path(args.db), source=args.source or None)
    print(f"Read {len(rows)} predictions from {args.db}")
    report = analyze(rows)
    write_report(report, Path(args.out))
    if report["with_gt"]:
        print(f"Overall top-1 accuracy: {report['overall_accuracy']:.4f}")
    print(f"Abstain rate: {report['abstain_rate']:.4f}")
    print(f"Mean latency: {report['latency_mean_ms']:.1f} ms")


if __name__ == "__main__":
    main()
