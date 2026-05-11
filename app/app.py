from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import predict_any


def discover_models() -> list[Path]:
    model_dir = ROOT / "models"
    if not model_dir.exists():
        return []
    return sorted(list(model_dir.glob("*.joblib")) + list(model_dir.glob("*.pt")))


st.set_page_config(page_title="Flower Recognition", page_icon=":cherry_blossom:", layout="centered")
st.title("Flower Recognition")

models = discover_models()
if not models:
    st.warning("No trained model found in models/.")
    st.stop()

selected_model = st.sidebar.selectbox("Model", models, format_func=lambda path: path.name)
top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=5)
uploaded_file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, use_container_width=True)
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        temp_path = Path(handle.name)

    try:
        predictions = predict_any(temp_path, selected_model, top_k=top_k)
        st.subheader("Predictions")
        for item in predictions:
            st.write(f"{item['label']} - {item['score']:.2%}")
            st.progress(min(max(item["score"], 0.0), 1.0))
    except Exception as exc:
        st.error(str(exc))
    finally:
        temp_path.unlink(missing_ok=True)

