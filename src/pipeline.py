from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.background_removal import cv2_available, is_likely_flower, remove_background
from src.embeddings import EmbeddingIndex
from src.feature_extraction import load_rgb_image
from src.inference import predict_any
from src.screening import ClassProfiles

# Füzyon skoru ağırlıkları (toplam = 1.0)
# CNN dokuyu zaten öğreniyor → model ağırlığı yüksek, doku düşük
FUSION_W_MODEL = 0.7
FUSION_W_COLOR = 0.25
FUSION_W_TEXTURE = 0.05

# Bu eşiğin altında en iyi tahmin "emin değilim" olarak işaretlenir.
# Tipik kalibre değer aralığı: 0.30–0.40
ABSTAIN_FUSION_THRESHOLD = 0.35

# Model softmax ile KNN softmax arasındaki harmanlama oranı (model_w + knn_w = 1.0)
KNN_RERANK_MODEL_WEIGHT = 0.7
KNN_RERANK_KNN_WEIGHT = 0.3


@dataclass
class ScreeningResult:
    label: str
    model_score: float         # model olasılığı (0–1)
    color_distance: float      # Bhattacharyya mesafesi
    color_percentile: float    # 0=en iyi, 100=en kötü (102 sınıf içinde)
    texture_distance: float    # Chi-kare mesafesi
    texture_percentile: float  # 0=en iyi, 100=en kötü
    verdict: str               # "olabilir" | "muhtemelen değil" | "kesinlikle bu olamaz"
    bg_removed: bool = True    # GrabCut maskesi kullanıldı mı
    fusion_score: float = 0.0  # 0.5*model + 0.3*renk + 0.2*doku
    abstain: bool = False      # True ise sistem bu tahminden emin değil

    @property
    def verdict_emoji(self) -> str:
        return {
            "olabilir": "🟢",
            "muhtemelen değil": "🟡",
            "kesinlikle bu olamaz": "🔴",
        }.get(self.verdict, "⚪")

    @property
    def color_match_pct(self) -> float:
        """Renk uyum yüzdesi: 100 = mükemmel, 0 = hiç uymuyor."""
        return max(0.0, 100.0 - self.color_percentile)

    @property
    def texture_match_pct(self) -> float:
        """Doku uyum yüzdesi: 100 = mükemmel, 0 = hiç uymuyor."""
        return max(0.0, 100.0 - self.texture_percentile)


def _fusion(model_score: float, color_pct: float, texture_pct: float) -> float:
    color_match = (100.0 - color_pct) / 100.0
    texture_match = (100.0 - texture_pct) / 100.0
    return (
        FUSION_W_MODEL * model_score
        + FUSION_W_COLOR * color_match
        + FUSION_W_TEXTURE * texture_match
    )


def _maybe_load_knn(
    embeddings_index: EmbeddingIndex | None,
    embeddings_path: str | Path | None,
    model_path: str | Path,
) -> EmbeddingIndex | None:
    """Embeddings index'ini cache'ten al veya diskten yükle. Yoksa None."""
    if embeddings_index is not None:
        return embeddings_index
    if embeddings_path is None:
        return None
    path = Path(embeddings_path)
    if not path.exists() or Path(model_path).suffix != ".pt":
        return None
    try:
        return EmbeddingIndex.load(path)
    except Exception as exc:
        print(f"[uyarı] Embeddings yüklenemedi ({path}): {exc}")
        return None


def _blend_knn(model_preds: list[dict], knn_probs: dict[str, float]) -> list[dict]:
    """
    Model softmax ile KNN softmax'ı ağırlıklı harmanlar, yeniden sıralar.
    KNN_RERANK_MODEL_WEIGHT × model + KNN_RERANK_KNN_WEIGHT × knn
    """
    blended = []
    for pred in model_preds:
        label = pred["label"]
        score = (
            KNN_RERANK_MODEL_WEIGHT * pred["score"]
            + KNN_RERANK_KNN_WEIGHT * knn_probs.get(label, 0.0)
        )
        blended.append({"label": label, "score": score})
    blended.sort(key=lambda p: p["score"], reverse=True)
    return blended


def predict_pipeline(
    image_path: str | Path,
    model_path: str | Path,
    profiles_path: str | Path | None = None,
    profiles: ClassProfiles | None = None,
    top_k: int = 3,
    image_size: int = 224,
    use_tta: bool = True,
    embeddings_path: str | Path | None = "models/train_embeddings.npz",
    embeddings_index: EmbeddingIndex | None = None,
    knn_k: int = 20,
    prefer_cuda: bool = True,
) -> tuple[list[ScreeningResult], np.ndarray, np.ndarray]:
    """
    Yeni pipeline akışı:
        1. Orijinal görüntü + GrabCut maskesi
        2. fg_ratio kontrolü — çiçek değilse skorlamayı orijinal üzerinden yap
        3. Renk + doku ön elemesi (102 sınıf), adaptive verdict ile survivor seti
        4. Model tüm dağılımı verir; survivor filtresi uygulanır (boşsa fallback)
        5. Füzyon skoru ile yeniden sıralama → top_k
    """
    image_path = Path(image_path)

    # Profil yolunu otomatik belirle: modelin yanındaki > genel
    if profiles_path is None:
        model_dir = Path(model_path).parent
        specific = model_dir / "class_profiles.npz"
        profiles_path = specific if specific.exists() else Path("models/class_profiles.npz")
    profiles_path = Path(profiles_path)

    # 1. Orijinal
    original_rgb = load_rgb_image(image_path, image_size=image_size)

    # 2. Maskeleme + çiçek tespiti
    masked_rgb, fg_mask = remove_background(image_path, image_size=image_size)
    use_masked = cv2_available() and is_likely_flower(fg_mask)
    scoring_img = masked_rgb if use_masked else original_rgb
    bg_removed = use_masked

    # 3. Profiller (cache veya disk)
    if profiles is None and profiles_path.exists():
        profiles = ClassProfiles.load(profiles_path)

    # 4. Tüm sınıflar için renk/doku skorları
    color_ranks = profiles.rank_color(scoring_img)
    texture_ranks = profiles.rank_texture(scoring_img)
    color_map = {cls: (dist, pct) for cls, dist, pct in color_ranks}
    texture_map = {cls: (dist, pct) for cls, dist, pct in texture_ranks}

    # 5. Survivor seti (adaptive verdict ile)
    survivors: set[str] = set()
    for cls in profiles.classes:
        c_dist, c_pct = color_map[cls]
        t_dist, t_pct = texture_map[cls]
        verdict = profiles.get_adaptive_verdict(
            cls, c_dist, t_dist,
            color_percentile=c_pct, texture_percentile=t_pct,
        )
        if verdict != "kesinlikle bu olamaz":
            survivors.add(cls)

    # 6. Model — tüm dağılımı al, survivor'lara filtre uygula
    n_classes = len(profiles.classes)
    model_preds = predict_any(
        image_path, model_path,
        top_k=n_classes, use_tta=use_tta, prefer_cuda=prefer_cuda,
    )

    # 6b. KNN reranking — embedding index varsa model skorunu harmanla
    knn_index = _maybe_load_knn(embeddings_index, embeddings_path, model_path)
    if knn_index is not None:
        try:
            knn_probs = knn_index.knn_class_softmax(
                image_path, model_path, k=knn_k, prefer_cuda=prefer_cuda,
            )
            model_preds = _blend_knn(model_preds, knn_probs)
        except Exception as exc:
            print(f"[uyarı] KNN reranking atlandı: {exc}")

    filtered = [p for p in model_preds if p["label"] in survivors] or model_preds

    # 7. Füzyon — biraz fazla aday üzerinde, sonra sırala
    consider = max(top_k * 3, 10)
    results: list[ScreeningResult] = []
    for pred in filtered[:consider]:
        label = pred["label"]
        c_dist, c_pct = color_map.get(label, (float("inf"), 100.0))
        t_dist, t_pct = texture_map.get(label, (float("inf"), 100.0))
        verdict = profiles.get_adaptive_verdict(
            label, c_dist, t_dist,
            color_percentile=c_pct, texture_percentile=t_pct,
        )
        fusion = _fusion(pred["score"], c_pct, t_pct)
        results.append(
            ScreeningResult(
                label=label,
                model_score=pred["score"],
                color_distance=c_dist,
                color_percentile=c_pct,
                texture_distance=t_dist,
                texture_percentile=t_pct,
                verdict=verdict,
                bg_removed=bg_removed,
                fusion_score=fusion,
                abstain=fusion < ABSTAIN_FUSION_THRESHOLD,
            )
        )

    results.sort(key=lambda r: r.fusion_score, reverse=True)
    return results[:top_k], original_rgb, masked_rgb
