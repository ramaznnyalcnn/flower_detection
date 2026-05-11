from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.classifiers import (
    DEFAULT_CLASSIFIERS,
    load_classical_bundle,
    save_classical_bundle,
    train_classifier,
    validate_classifier_names,
)
from src.data_utils import (
    class_distribution,
    class_names_from_records,
    collect_image_records,
    ensure_records,
    labels_from_records,
    limit_records_per_class,
    load_dataset_splits,
    write_json,
)
from src.dimensionality import compute_embedding, save_embedding_plot
from src.evaluation import evaluate_estimator, evaluate_predictions, write_markdown_report, write_metrics_summary
from src.feature_extraction import DEFAULT_FEATURES, FeatureExtractor, validate_feature_names
from src.inference import predict_any


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def records_for_split(args) -> list:
    data_dir = Path(args.data_dir)
    if args.split == "all":
        split_dirs = [data_dir / name for name in ("train", "val", "test")]
        if all(path.exists() for path in split_dirs):
            records = []
            for split_dir in split_dirs:
                records.extend(collect_image_records(split_dir))
        else:
            records = collect_image_records(data_dir)
        return limit_records_per_class(records, args.limit_per_class, random_state=args.random_state)

    split_dir = data_dir / args.split
    if split_dir.exists():
        records = collect_image_records(split_dir)
    else:
        records = collect_image_records(data_dir)
    return limit_records_per_class(records, args.limit_per_class, random_state=args.random_state)


def validate_data(args) -> None:
    splits = load_dataset_splits(
        args.data_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )
    summary = {}
    for split, records in splits.items():
        summary[split] = {
            "samples": len(records),
            "classes": len(class_distribution(records)),
            "class_distribution": class_distribution(records),
        }
        print(f"{split}: {summary[split]['samples']} images, {summary[split]['classes']} classes")

    output_path = Path(args.results_dir) / "data_validation.json"
    write_json(output_path, summary)
    print(f"Wrote {output_path}")


def run_classical_baseline(args) -> None:
    feature_types = validate_feature_names(args.features)
    classifier_names = validate_classifier_names(args.classifiers)
    splits = load_dataset_splits(
        args.data_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )
    splits = {
        split: limit_records_per_class(records, args.limit_per_class, random_state=args.random_state)
        for split, records in splits.items()
    }
    ensure_records(splits["train"], "training split")

    extractor = FeatureExtractor(feature_types=feature_types, image_size=args.image_size)
    output_dir = Path(args.results_dir) / "classical_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Features: {', '.join(feature_types)}")
    print("Extracting train features")
    x_train = extractor.transform_records(splits["train"])
    y_train = labels_from_records(splits["train"])

    split_features = {}
    for split in ("val", "test"):
        if splits.get(split):
            print(f"Extracting {split} features")
            split_features[split] = (
                extractor.transform_records(splits[split]),
                labels_from_records(splits[split]),
            )

    rows = []
    all_classes = class_names_from_records(splits["train"])
    for classifier_name in classifier_names:
        print(f"Training {classifier_name}")
        estimator = train_classifier(classifier_name, x_train, y_train, random_state=args.random_state)
        model_rows = []

        for split, (x_split, y_split) in split_features.items():
            metrics = evaluate_estimator(
                estimator,
                x_split,
                y_split,
                classes=all_classes,
                output_dir=output_dir / classifier_name,
                prefix=split,
            )
            row = {"model": classifier_name, "split": split, **metrics}
            rows.append(row)
            model_rows.append(row)
            print(f"{classifier_name} {split}: accuracy={metrics['accuracy']:.4f}")

        model_path = models_dir / f"classical_{classifier_name}.joblib"
        save_classical_bundle(
            model_path,
            estimator=estimator,
            classifier_name=classifier_name,
            feature_types=feature_types,
            image_size=args.image_size,
            classes=list(getattr(estimator, "classes_", all_classes)),
            metrics={"rows": model_rows},
        )
        print(f"Saved {model_path}")

    write_metrics_summary(output_dir / "metrics_summary.csv", rows)
    write_markdown_report(Path(args.results_dir) / "REPORT.md", "Flower Recognition Experiment Report", rows)
    print(f"Wrote {output_dir / 'metrics_summary.csv'}")


def evaluate_classical(args) -> None:
    if not args.model_path:
        raise ValueError("--model-path is required for evaluate")

    bundle = load_classical_bundle(args.model_path)
    records = records_for_split(args)
    ensure_records(records, f"{args.split} split")
    extractor = FeatureExtractor(
        feature_types=tuple(bundle["feature_types"]),
        image_size=int(bundle.get("image_size", args.image_size)),
    )
    x = extractor.transform_records(records)
    y = labels_from_records(records)

    model_name = Path(args.model_path).stem
    output_dir = Path(args.results_dir) / "evaluation" / model_name
    metrics = evaluate_estimator(
        bundle["estimator"],
        x,
        y,
        classes=list(bundle.get("classes", sorted(set(y)))),
        output_dir=output_dir,
        prefix=args.split,
    )
    rows = [{"model": model_name, "split": args.split, **metrics}]
    write_metrics_summary(output_dir / "metrics_summary.csv", rows)
    print(f"{model_name} {args.split}: accuracy={metrics['accuracy']:.4f}")


def evaluate_cnn(args) -> None:
    if not args.model_path:
        raise ValueError("--model-path is required for evaluate")

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    from src.cnn_model import build_transforms, load_cnn_checkpoint

    model, checkpoint, device = load_cnn_checkpoint(args.model_path, prefer_cuda=not args.cpu)
    data_dir = Path(args.data_dir)
    eval_dir = data_dir / args.split if (data_dir / args.split).exists() else data_dir
    dataset = datasets.ImageFolder(eval_dir, transform=build_transforms(checkpoint.get("image_size", 224), train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true = []
    y_pred = []
    probabilities = []
    classes = list(checkpoint["classes"])
    start = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            pred_indices = probs.argmax(axis=1)
            probabilities.extend(probs)
            y_pred.extend(classes[index] for index in pred_indices)
            y_true.extend(dataset.classes[index] for index in targets.numpy())
    elapsed = time.perf_counter() - start

    model_name = Path(args.model_path).stem
    output_dir = Path(args.results_dir) / "evaluation" / model_name
    metrics = evaluate_predictions(
        y_true,
        y_pred,
        probabilities=np.asarray(probabilities),
        classes=classes,
        output_dir=output_dir,
        prefix=args.split,
        elapsed_seconds=elapsed,
    )
    rows = [{"model": model_name, "split": args.split, **metrics}]
    write_metrics_summary(output_dir / "metrics_summary.csv", rows)
    print(f"{model_name} {args.split}: accuracy={metrics['accuracy']:.4f}")


def run_evaluate(args) -> None:
    if Path(args.model_path).suffix == ".pt":
        evaluate_cnn(args)
    else:
        evaluate_classical(args)


def run_cnn_train(args) -> None:
    from src.cnn_model import train_cnn

    output_path = Path(args.models_dir) / "resnet50_best.pt"
    summary = train_cnn(
        data_dir=args.data_dir,
        output_path=output_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        pretrained=not args.no_pretrained,
        fine_tune=args.fine_tune,
        prefer_cuda=not args.cpu,
        num_workers=args.num_workers,
    )
    print(f"Saved {output_path}")
    print(f"Best val accuracy: {summary['best_val_accuracy']:.4f}")


def run_predict(args) -> None:
    if not args.model_path:
        raise ValueError("--model-path is required for predict")
    if not args.image:
        raise ValueError("--image is required for predict")

    predictions = predict_any(args.image, args.model_path, top_k=args.top_k)
    for index, item in enumerate(predictions, start=1):
        print(f"{index}. {item['label']}: {item['score']:.4f}")


def run_visualize(args) -> None:
    feature_types = validate_feature_names(args.features)
    records = records_for_split(args)
    ensure_records(records, f"{args.split} split")
    extractor = FeatureExtractor(feature_types=feature_types, image_size=args.image_size)
    x = extractor.transform_records(records)
    y = labels_from_records(records)
    embedding = compute_embedding(x, y=y, method=args.embedding_method, random_state=args.random_state)
    output_path = Path(args.results_dir) / "visualizations" / f"{args.embedding_method}_{args.split}.png"
    save_embedding_plot(embedding, y, output_path, title=f"{args.embedding_method.upper()} - {args.split}")
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower recognition project runner")
    parser.add_argument(
        "--task",
        required=True,
        choices=["validate-data", "classical-baseline", "cnn-train", "evaluate", "predict", "visualize"],
    )
    parser.add_argument("--data-dir", default="data/processed/oxford102")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--classifiers", nargs="+", default=list(DEFAULT_CLASSIFIERS))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-method", default="pca", choices=["pca", "lda", "tsne", "umap"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handlers = {
        "validate-data": validate_data,
        "classical-baseline": run_classical_baseline,
        "cnn-train": run_cnn_train,
        "evaluate": run_evaluate,
        "predict": run_predict,
        "visualize": run_visualize,
    }
    handlers[args.task](args)


if __name__ == "__main__":
    main()
