from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


DEFAULT_CLASSIFIERS = ("svm_rbf", "random_forest")
AVAILABLE_CLASSIFIERS = ("svm_rbf", "random_forest", "knn", "naive_bayes", "xgboost")


class XGBoostLabelAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, x, y):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency branch
            raise ImportError("xgboost is required for the xgboost classifier.") from exc

        self.encoder_ = LabelEncoder()
        encoded_y = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.random_state,
        )
        self.model_.fit(x, encoded_y)
        return self

    def predict(self, x):
        encoded = self.model_.predict(x)
        return self.encoder_.inverse_transform(encoded.astype(int))

    def predict_proba(self, x):
        return self.model_.predict_proba(x)


def validate_classifier_names(classifier_names) -> tuple[str, ...]:
    classifier_names = tuple(classifier_names)
    invalid = [name for name in classifier_names if name not in AVAILABLE_CLASSIFIERS]
    if invalid:
        raise ValueError(f"Unknown classifier(s): {', '.join(invalid)}")
    return classifier_names


def make_classifier(name: str, random_state: int = 42) -> Pipeline:
    if name == "svm_rbf":
        estimator = SVC(
            C=10,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([("model", estimator)])

    if name == "knn":
        return Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))])

    if name == "naive_bayes":
        return Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())])

    if name == "xgboost":
        return Pipeline([("model", XGBoostLabelAdapter(random_state=random_state))])

    raise ValueError(f"Unknown classifier: {name}")


def train_classifier(name: str, x_train: np.ndarray, y_train: list[str], random_state: int = 42) -> Pipeline:
    classifier = make_classifier(name, random_state=random_state)
    classifier.fit(x_train, y_train)
    return classifier


def save_classical_bundle(
    path: str | Path,
    estimator,
    classifier_name: str,
    feature_types: tuple[str, ...],
    image_size: int,
    classes: list[str],
    metrics: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "classical",
        "classifier_name": classifier_name,
        "estimator": estimator,
        "feature_types": tuple(feature_types),
        "image_size": image_size,
        "classes": list(classes),
        "metrics": metrics or {},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(payload, path)


def load_classical_bundle(path: str | Path) -> dict:
    payload = joblib.load(path)
    if payload.get("model_type") != "classical":
        raise ValueError(f"Not a classical model bundle: {path}")
    return payload

