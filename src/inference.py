from __future__ import annotations

from pathlib import Path

import numpy as np

from src.classifiers import load_classical_bundle
from src.cnn_model import predict_cnn
from src.feature_extraction import FeatureExtractor


def predict_classical(image_path: str | Path, model_path: str | Path, top_k: int = 5) -> list[dict]:
    bundle = load_classical_bundle(model_path)
    extractor = FeatureExtractor(
        feature_types=tuple(bundle["feature_types"]),
        image_size=int(bundle.get("image_size", 224)),
    )
    x = extractor.transform_path(image_path).reshape(1, -1)
    estimator = bundle["estimator"]

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x)[0]
        classes = np.asarray(getattr(estimator, "classes_", bundle.get("classes", [])))
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        return [
            {"label": str(classes[index]), "score": float(probabilities[index])}
            for index in top_indices
        ]

    prediction = estimator.predict(x)[0]
    return [{"label": str(prediction), "score": 1.0}]


def predict_any(image_path: str | Path, model_path: str | Path, top_k: int = 5) -> list[dict]:
    model_path = Path(model_path)
    if model_path.suffix == ".pt":
        return predict_cnn(image_path, model_path, top_k=top_k)
    return predict_classical(image_path, model_path, top_k=top_k)

