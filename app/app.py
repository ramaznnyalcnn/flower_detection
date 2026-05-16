from __future__ import annotations

import io
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.background_removal import cv2_available
from src.embeddings import EmbeddingIndex
from src.inference import predict_any
from src.pipeline import predict_pipeline
from src.screening import ClassProfiles
try:
    from src.flower_detector import is_flower as _clip_is_flower
    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False

PROFILES_PATH = ROOT / "models" / "class_profiles.npz"
EMBEDDINGS_PATH = ROOT / "models" / "train_embeddings.npz"


def _model_profiles_path(model_path: Path) -> Path:
    """Modele özel profil varsa onu, yoksa genel profili döndür."""
    specific = model_path.parent / "class_profiles.npz"
    return specific if specific.exists() else PROFILES_PATH


def _model_embeddings_path(model_path: Path) -> Path:
    """Modele özel embeddings varsa onu, yoksa genel embeddings döndür."""
    specific = model_path.parent / "train_embeddings.npz"
    return specific if specific.exists() else EMBEDDINGS_PATH


@st.cache_resource
def _model_num_classes(path_str: str) -> int:
    try:
        import torch
        ckpt = torch.load(path_str, map_location="cpu", weights_only=False)
        return len(ckpt.get("classes", []))
    except Exception:
        return 0


@st.cache_resource
def _load_profiles(path_str: str) -> ClassProfiles | None:
    """Profil dosyasını bir kez yükle, Streamlit oturumu boyunca önbelleğe al."""
    path = Path(path_str)
    if not path.exists():
        return None
    return ClassProfiles.load(path)


@st.cache_resource
def _load_embeddings(path_str: str) -> EmbeddingIndex | None:
    """Embeddings indexini bir kez yükle (varsa)."""
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return EmbeddingIndex.load(path)
    except Exception:
        return None


TURKISH_NAMES = {
    "gul": "Gül", "lale": "Lale", "papatya": "Papatya",
    "aysicegi": "Ayçiçeği", "karahindiba": "Karahindiba", "lilyum": "Lilyum",
    "orkide": "Orkide", "kasimpati": "Kasımpatı", "bougainvillea": "Bougainvillea",
    "iris": "İris", "menekse": "Menekşe", "sardunya": "Sardunya",
    "karanfil": "Karanfil", "gelincik": "Gelincik", "sumbul": "Sümbül",
    "nergis": "Nergis", "manolya": "Manolya", "leylak": "Leylak",
}

def _display_name(label: str) -> str:
    return TURKISH_NAMES.get(label, label.replace("_", " ").title())


def discover_models() -> list[Path]:
    model_dir = ROOT / "models"
    if not model_dir.exists():
        return []
    pts = [p for p in model_dir.rglob("*.pt") if "best" not in p.stem or "v2" in p.stem]
    joblibs = list(model_dir.rglob("*.joblib"))
    return sorted(pts + joblibs)

MODEL_LABELS = {
    "resnet50_turkey_v2.pt": "🌸 Türkiye Çiçekleri v2 (web verisi)",
    "resnet50_turkey.pt":    "🌸 Türkiye Çiçekleri v1",
    "resnet50_v2.pt":        "🌍 Oxford 102 Sınıf",
    "resnet50_best.pt":      "🌍 Oxford 102 Sınıf (eski)",
    "classical_random_forest.joblib": "🌲 Random Forest",
    "classical_svm_rbf.joblib":       "📐 SVM",
}

def _model_label(p: Path) -> str:
    return MODEL_LABELS.get(p.name, p.name)


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
    "Model", models, format_func=_model_label
)
active_profiles_path = _model_profiles_path(selected_model)
active_embeddings_path = _model_embeddings_path(selected_model)
profiles_available = active_profiles_path.exists()
model_n_classes = _model_num_classes(str(selected_model))
profile_n_classes = 0
if profiles_available:
    try:
        import numpy as np
        _tmp = np.load(active_profiles_path, allow_pickle=True)
        profile_n_classes = len(_tmp["classes"])
    except Exception:
        pass

profile_mismatch = profiles_available and profile_n_classes != model_n_classes and model_n_classes > 0
pipeline_ok = profiles_available and not profile_mismatch

use_pipeline = st.sidebar.toggle(
    "Akıllı Pipeline (ön eleme + karar)",
    value=pipeline_ok,
    disabled=not pipeline_ok,
    help="Renk + doku ön eleme ile 3 katmanlı analiz" if pipeline_ok
    else "Profil dosyası bu model için uygun değil.",
)
if profile_mismatch:
    st.sidebar.warning(
        f"⚠️ Profil ({profile_n_classes} sınıf) ↔ Model ({model_n_classes} sınıf) uyumsuz.\n"
        "Pipeline devre dışı. Klasik mod kullanılıyor."
    )
elif not profiles_available:
    st.sidebar.warning(
        "class_profiles.npz bulunamadı.\n"
        "`python scripts/build_class_profiles.py`"
    )
if not cv2_available():
    st.sidebar.warning(
        "⚠️ Arka plan kaldırma devre dışı.\n"
        "Etkinleştirmek için:\n`pip install opencv-python`"
    )

# GPU durum bilgisi
try:
    import torch
    if torch.cuda.is_available():
        st.sidebar.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.info("💻 CPU modunda çalışıyor (CUDA bulunamadı)")
except ImportError:
    st.sidebar.info("💻 PyTorch yüklü değil")

top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=3)

if CLIP_AVAILABLE:
    use_clip_filter = st.sidebar.toggle(
        "🌸 Çiçek tespiti (CLIP)", value=True,
        help="Resim yüklenince önce çiçek mi diye CLIP ile kontrol eder.",
    )
else:
    use_clip_filter = False

tab_upload, tab_paste, tab_url, tab_search = st.tabs(
    ["📁 Dosya / Sürükle", "📋 Yapıştır", "🔗 URL", "🔍 Google Arama"]
)

with tab_upload:
    uploaded_file = st.file_uploader(
        "Sürükle bırak veya seç", type=["jpg", "jpeg", "png", "webp"]
    )

with tab_paste:
    st.markdown("**Ctrl+V** ile panodaki resmi yapıştır:")
    try:
        from streamlit_paste_button import paste_image_button
        paste_result = paste_image_button("📋 Buraya Yapıştır (Ctrl+V)", key="paste_btn")
        pasted_image = paste_result.image_data if paste_result and paste_result.image_data else None
    except Exception:
        pasted_image = None
        st.info("Yapıştır özelliği yüklenemedi.")

with tab_url:
    url_input = st.text_input("Resim URL'si", placeholder="https://example.com/flower.jpg")
    url_submit = st.button("Tahmin Et", key="url_btn")

with tab_search:
    search_query = st.text_input("Çiçek adı", placeholder="örn: rose, sunflower, lavender...")
    search_submit = st.button("Ara ve Tahmin Et", key="search_btn")



def _fetch_image_from_url(url: str) -> Path:
    """URL'den resim indir, geçici dosyaya kaydet."""
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tmp.name, "JPEG")
    return Path(tmp.name)


def _google_search_image(query: str) -> Path:
    """Google Images'tan ilk sonucu çek."""
    import urllib.parse, re
    q = urllib.parse.quote(f"{query} flower close up")
    search_url = f"https://www.google.com/search?q={q}&tbm=isch&tbs=isz:m"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(search_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    # img src içindeki ilk gerçek resmi bul
    urls = re.findall(r'"(https://[^"]+\.(?:jpg|jpeg|png|webp))"', html)
    urls = [u for u in urls if "gstatic" not in u and "google" not in u]
    if not urls:
        raise ValueError(f"'{query}' için sonuç bulunamadı.")
    return _fetch_image_from_url(urls[0])


# Aktif girdi kaynağını belirle
active_image_path: Path | None = None
active_image_label: str = ""

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        active_image_path = Path(handle.name)
    active_image_label = uploaded_file.name

elif pasted_image is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pasted_image.save(tmp.name)
    active_image_path = Path(tmp.name)
    active_image_label = "Panodan yapıştırıldı"

elif url_submit and url_input:
    with st.spinner("URL'den resim indiriliyor..."):
        try:
            active_image_path = _fetch_image_from_url(url_input)
            active_image_label = url_input
        except Exception as e:
            st.error(f"URL'den resim alınamadı: {e}")

elif search_submit and search_query:
    with st.spinner(f"'{search_query}' Google'da aranıyor..."):
        try:
            active_image_path = _google_search_image(search_query)
            active_image_label = f"Google: {search_query}"
        except Exception as e:
            st.error(f"Arama başarısız: {e}")

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
if active_image_path is not None:
    temp_path = active_image_path
    try:
        # ---- CLIP çiçek-değil ön filtresi ----
        if use_clip_filter and CLIP_AVAILABLE:
            with st.spinner("Resim çiçek mi kontrol ediliyor..."):
                is_flw, score = _clip_is_flower(temp_path, threshold=0.10)
            if not is_flw:
                st.image(temp_path, width=400)
                st.error(
                    f"🚫 Bu görselin çiçek olmadığı tespit edildi (CLIP skoru: {score:+.3f}). "
                    f"Lütfen bir çiçek resmi yükleyin."
                )
                st.stop()

        # ---- Pipeline modu ----
        if use_pipeline:
            cached_profiles = _load_profiles(str(active_profiles_path))
            cached_embeddings = _load_embeddings(str(active_embeddings_path))
            with st.spinner("Arka plan kaldırılıyor ve analiz yapılıyor..."):
                results, original_rgb, masked_rgb = predict_pipeline(
                    temp_path,
                    selected_model,
                    profiles_path=active_profiles_path,
                    profiles=cached_profiles,
                    embeddings_path=active_embeddings_path,
                    embeddings_index=cached_embeddings,
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
            if results and results[0].abstain:
                st.warning(
                    f"⚠️ Sistem bu görselden emin değil. "
                    f"Görsel bilinen {model_n_classes} sınıftan biri olmayabilir "
                    f"veya görüntü kalitesi yetersiz."
                )
            st.subheader(f"Top {top_k} Tahmin")

            for rank, res in enumerate(results, start=1):
                with st.container(border=True):
                    header_col, badge_col = st.columns([3, 2])
                    with header_col:
                        st.markdown(f"### {rank}. {_display_name(res.label)}")
                        st.progress(res.model_score, text=f"Doğruluk: **%{res.model_score*100:.1f}**")
                    with badge_col:
                        st.markdown(
                            _verdict_badge(res.verdict), unsafe_allow_html=True
                        )

                    f_col, m_col, c_col = st.columns(3)
                    with f_col:
                        st.metric("Füzyon", f"{res.fusion_score:.1%}")
                    with m_col:
                        st.metric("Model Skoru", f"{res.model_score:.1%}")
                    with c_col:
                        st.metric(
                            "Renk Uyumu",
                            f"{res.color_match_pct:.0f}%",
                            delta=f"{'✓' if res.color_percentile <= 60 else '✗'}",
                            delta_color="normal" if res.color_percentile <= 60 else "inverse",
                        )

                    _color_bar("Renk eşleşme", res.color_match_pct, res.color_percentile)

        # ---- Klasik mod ----
        else:
            st.image(temp_path, use_container_width=False, width=400)
            with st.spinner("Tahmin yapılıyor..."):
                predictions = predict_any(temp_path, selected_model, top_k=top_k)
            st.subheader("Tahminler")
            for item in predictions:
                name = _display_name(item['label'])
                pct = item['score']
                st.progress(min(max(pct, 0.0), 1.0), text=f"**{name}** — %{pct*100:.1f}")

    except Exception as exc:
        st.error(str(exc))
        raise
    finally:
        temp_path.unlink(missing_ok=True)
