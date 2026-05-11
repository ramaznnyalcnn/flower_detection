# 🌸 Çiçek Türü Tanıma — Pattern Recognition Projesi

Klasik örüntü tanıma yöntemleri ile derin öğrenmeyi karşılaştıran, çok veri setli, gerçek dünya dayanıklılığı odaklı bir çiçek sınıflandırma çalışması.

---

## Projenin Amacı

Bu proje yalnızca "iyi doğruluk oranı elde etmek" değil, **modelin gerçek dünyada çalışıp çalışmadığını** kanıtlamayı hedefler.

**İki aşamalı eğitim stratejisi:**
1. **Oxford Flowers 102** ile temiz, kaliteli veriyle eğitim
2. **Web'den toplanan gerçek dünya görselleri** ile ikinci eğitim — farklı boyutlar, kaliteler, arka planlar

Temel soru: Temiz veriyle eğitilen model, telefondan çekilen rastgele bir çiçek fotoğrafını tanıyabilir mi?

---

## Proje Yapısı

```
flower_recognition/
│
├── download_oxford102.py         # Oxford 102 veri setini indir
├── download_google_images.py     # Web'den ek görsel topla
├── clean_data.py                 # Veri temizleme pipeline'ı
├── requirements.txt              # Bağımlılıklar
│
├── src/
│   ├── data_utils.py            # Dataset keşfi, split ve sınıf dağılımları
│   ├── feature_extraction.py     # HOG, LBP, Color, GLCM özellikleri
│   ├── classifiers.py            # SVM, k-NN, RF, XGBoost, Bayes
│   ├── cnn_model.py              # CNN (Transfer Learning + CUDA)
│   ├── evaluation.py             # Metrikler, ROC, confusion matrix
│   ├── dimensionality.py         # PCA, LDA, t-SNE, UMAP
│   └── inference.py              # Tek görsel tahmini
│
├── data/
│   ├── raw/
│   │   ├── oxford102/            # Oxford Flowers 102
│   │   └── google_images/        # Web'den toplanan görseller
│   └── processed/                # Temizlenmiş ve normalize edilmiş
│
├── models/                       # Eğitilmiş modeller
├── results/                      # Sonuçlar, grafikler, tablolar
│
└── app/
    └── app.py                    # Streamlit web arayüzü
```

---

## Kullanılan Yöntemler

### Feature Extraction (Özellik Çıkarımı)
| Grup | Özellikler |
|------|-----------|
| Renk | HSV histogram, renk momentleri, dominant renk |
| Doku | LBP, GLCM, Gabor filter |
| Şekil | HOG, Hu Moments |

### Sınıflandırıcılar
- SVM (Linear + RBF kernel)
- k-NN
- Naive Bayes
- Random Forest
- XGBoost
- CNN (ResNet/EfficientNet — Transfer Learning, CUDA)

### Boyut İndirgeme
- PCA
- LDA
- t-SNE
- UMAP

---

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
# 1. Oxford 102 indir
python download_oxford102.py

# 2. Web'den ek görsel topla
python download_google_images.py --limit 50

# 3. Veriyi normalize et
# Oxford benchmark için görsel elemeden çalıştırmak daha iyi baseline verir.
python clean_data.py --source oxford --keep-all

# 4. Veri yapısını doğrula
python run.py --task validate-data

# 5. Oxford resmi split yerine 70/15/15 split hazırla
python scripts/prepare_stratified_split.py \
  --input-dir data/processed/oxford102 \
  --output-dir data/processed/oxford102_70_15_15 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --seed 42 \
  --overwrite

# 6. İlk klasik baseline (HSV + HOG, Random Forest hızlı baseline)
python run.py --task classical-baseline \
  --data-dir data/processed/oxford102_70_15_15 \
  --classifiers random_forest

# Alternatif: resmi split üzerinde klasik baseline
python run.py --task classical-baseline

# 7. CUDA ile CNN transfer learning
# RTX 3050 4GB gibi GPU'larda batch-size 16 daha güvenlidir.
python run.py --task cnn-train \
  --data-dir data/processed/oxford102_70_15_15 \
  --epochs 10 \
  --batch-size 16

# 8. Eğitilmiş modeli değerlendir
python run.py --task evaluate --model-path models/classical_svm_rbf.joblib

# 9. Tek görsel tahmini
python run.py --task predict --model-path models/classical_svm_rbf.joblib --image path/to/flower.jpg

# 10. Web arayüzü
streamlit run app/app.py
```

## Deney Çıktıları

- `models/`: eğitilmiş klasik model paketleri ve CNN checkpointleri
- `results/classical_baseline/`: metrik JSON dosyaları, classification report ve confusion matrix görselleri
- `results/metrics_summary.csv` veya alt klasörlerdeki summary dosyaları: model karşılaştırma tabloları
- `results/REPORT.md`: rapor iskeleti ve son deney özeti
