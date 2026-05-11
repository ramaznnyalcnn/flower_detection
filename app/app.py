from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.background_removal import cv2_available
from src.inference import predict_any
from src.pipeline import predict_pipeline

PROFILES_PATH = ROOT / "models" / "class_profiles.npz"


def discover_models() -> list[Path]:
    model_dir = ROOT / "models"
    if not model_dir.exists():
        return []
    return sorted(list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.pt")))


# ------------------------------------------------------------------
# Sayfa ayarı
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Flower Recognition",
    page_icon=":cherry_blossom:",
    layout="wide",
)
st.title(":cherry_blossom: Çiçek Tanıma")

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
models = discover_models()
if not models:
    st.warning("models/ klasöründe eğitilmiş model bulunamadı.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Model", models, format_func=lambda p: p.name
)
profiles_available = PROFILES_PATH.exists()
use_pipeline = st.sidebar.toggle(
    "Akıllı Pipeline (ön eleme + karar)",
    value=profiles_available,
    disabled=not profiles_available,
    help="Renk ve doku ön elemeli pipeline. Profil dosyası gerektirir."
    if not profiles_available
    else "Renk + doku ön eleme ile 3 katmanlı analiz",
)
if not profiles_available:
    st.sidebar.warning(
        "class_profiles.npz bulunamadı. Pipeline kullanmak için:\n"
        "`python scripts/build_class_profiles.py`"
    )
if not cv2_available():
    st.sidebar.warning(
        "⚠️ Arka plan kaldırma devre dışı.\n"
        "Etkinleştirmek için:\n`pip install opencv-python`"
    )

top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=3)

uploaded_file = st.file_uploader(
    "Çiçek görüntüsü yükle", type=["jpg", "jpeg", "png", "webp"]
)

# ------------------------------------------------------------------
# Yardımcı render fonksiyonları
# ------------------------------------------------------------------

def _color_bar(label: str, match_pct: float, percentile: float) -> None:
    """0-100 arası uyum yüzdesi için renkli progress bar."""
    if percentile > 80:
        color = "#e74c3c"   # kırmızı
    elif percentile > 60:
        color = "#f39c12"   # turuncu
    else:
        color = "#27ae60"   # yeşil
    st.markdown(
        f"**{label}:** "
        f'<span style="color:{color}; font-weight:bold">{match_pct:.0f}%</span>',
        unsafe_allow_html=True,
    )
    st.progress(int(match_pct))


def _verdict_badge(verdict: str) -> str:
    colors = {
        "olabilir": ("#27ae60", "#d5f5e3"),
        "muhtemelen değil": ("#d68910", "#fef9e7"),
        "kesinlikle bu olamaz": ("#c0392b", "#fadbd8"),
    }
    emojis = {
        "olabilir": "🟢",
        "muhtemelen değil": "🟡",
        "kesinlikle bu olamaz": "🔴",
    }
    fg, bg = colors.get(verdict, ("#555", "#eee"))
    emoji = emojis.get(verdict, "⚪")
    return (
        f'<span style="background:{bg}; color:{fg}; padding:4px 10px; '
        f'border-radius:12px; font-weight:bold; font-size:0.9em">'
        f'{emoji} {verdict}</span>'
    )


# ------------------------------------------------------------------
# Ana tahmin akışı
# ------------------------------------------------------------------
if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        temp_path = Path(handle.name)

    try:
        # ---- Pipeline modu ----
        if use_pipeline:
            with st.spinner("Arka plan kaldırılıyor ve analiz yapılıyor..."):
                results, original_rgb, masked_rgb = predict_pipeline(
                    temp_path,
                    selected_model,
                    profiles_path=PROFILES_PATH,
                    top_k=top_k,
                )

            col_orig, col_masked = st.columns(2)
            with col_orig:
                st.subheader("Orijinal")
                st.image(original_rgb, use_container_width=True)
            with col_masked:
                st.subheader("Arka Plan Kaldırılmış")
                st.image(masked_rgb, use_container_width=True)

            st.divider()
            st.subheader(f"Top {top_k} Tahmin")

            for rank, res in enumerate(results, start=1):
                with st.container(border=True):
                    header_col, badge_col = st.columns([3, 2])
                    with header_col:
                        st.markdown(f"### {rank}. {res.label.replace('_', ' ').title()}")
                    with badge_col:
                        st.markdown(
                            _verdict_badge(res.verdict), unsafe_allow_html=True
                        )

                    m_col, c_col, t_col = st.columns(3)
                    with m_col:
                        st.metric("Model Skoru", f"{res.model_score:.1%}")
                    with c_col:
                        st.metric(
                            "Renk Uyumu",
                            f"{res.color_match_pct:.0f}%",
                            delta=f"{'✓' if res.color_percentile <= 60 else '✗'}",
                            delta_color="normal" if res.color_percentile <= 60 else "inverse",
                        )
                    with t_col:
                        st.metric(
                            "Doku Uyumu",
                            f"{res.texture_match_pct:.0f}%",
                            delta=f"{'✓' if res.texture_percentile <= 60 else '✗'}",
                            delta_color="normal" if res.texture_percentile <= 60 else "inverse",
                        )

                    bar_col1, bar_col2 = st.columns(2)
                    with bar_col1:
                        _color_bar("Renk eşleşme", res.color_match_pct, res.color_percentile)
                    with bar_col2:
                        _color_bar("Doku eşleşme", res.texture_match_pct, res.texture_percentile)

        # ---- Klasik mod ----
        else:
            st.image(uploaded_file, use_container_width=False, width=400)
            with st.spinner("Tahmin yapılıyor..."):
                predictions = predict_any(temp_path, selected_model, top_k=top_k)
            st.subheader("Tahminler")
            for item in predictions:
                st.write(f"**{item['label']}** — {item['score']:.2%}")
                st.progress(min(max(item["score"], 0.0), 1.0))

    except Exception as exc:
        st.error(str(exc))
        raise
    finally:
        temp_path.unlink(missing_ok=True)
