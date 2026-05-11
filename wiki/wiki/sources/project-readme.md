---
title: Proje README ve Mevcut Kod Tabanı
category: source
summary: Çiçek tanıma projesinin başlangıç dökümanı — amaç, yapı, kurulum ve şu ana kadar yazılmış scriptler.
tags: [project, readme, baseline]
sources: 1
source_path: README.md
source_date: 2026-05
authors: [ramazan]
ingested: 2026-05-11
updated: 2026-05-11
---

# Proje README ve Mevcut Kod Tabanı

## TL;DR

Klasik örüntü tanıma yöntemleri (HOG, LBP, SVM, k-NN, RF, XGBoost) ile derin öğrenmeyi (CNN, transfer learning) karşılaştıran çiçek sınıflandırma projesi. Temel hipotez: temiz veriyle (Oxford Flowers 102) eğitilen model, gerçek dünya görsellerinde (web'den toplanan, çeşitli kalite ve arka plan) ne kadar dayanıklı olur?

## Temel Tasarım Kararları

- **İki aşamalı eğitim:** Önce [[entities/oxford-flowers-102]] ile temiz veri, sonra web görselleriyle gerçek dünya transferi.
- **Karşılaştırmalı yaklaşım:** [[concepts/classical-classifiers]] vs [[concepts/deep-learning-approach]] — tek bir "en iyi" model değil, trade-off matrisi.
- **Pipeline odaklı:** Veri toplama → temizleme → özellik çıkarımı → sınıflandırma → değerlendirme → arayüz.

## Mevcut Durum (2026-05-11)

| Bileşen | Durum | Dosya |
|---------|-------|-------|
| Oxford 102 indirme | ✅ Hazır | `download_oxford102.py` (144 satır) |
| Web görsel toplama | ✅ Hazır | `download_google_images.py` (247 satır, bing-image-downloader) |
| Veri temizleme | ✅ Hazır | `clean_data.py` (354 satır) — [[concepts/data-cleaning-pipeline]] |
| Feature extraction | ❌ Yok | `src/feature_extraction.py` planda |
| Klasik sınıflandırıcılar | ❌ Yok | `src/classifiers.py` planda |
| CNN modeli | ❌ Yok | `src/cnn_model.py` planda |
| Değerlendirme | ❌ Yok | `src/evaluation.py` planda |
| Boyut indirgeme | ❌ Yok | `src/dimensionality.py` planda |
| Eğitim entry-point | ❌ Yok | `run.py` planda |
| Web arayüzü | ❌ Yok | `app/app.py` planda (Streamlit) |

## Bağımlılıklar (özet)

- **Temel:** numpy, scipy, Pillow, scikit-learn, matplotlib, seaborn
- **Görüntü işleme:** opencv-python, scikit-image, imagehash
- **Klasik ML:** xgboost
- **Deep learning:** torch, torchvision (CUDA hedefli)
- **Veri toplama:** bing-image-downloader
- **Boyut indirgeme:** umap-learn
- **Arayüz:** streamlit, gradio

## Bilinen Açık Konular

- `{data` isimli garip bir dosya/klasör projede mevcut — muhtemelen kazara oluşmuş, temizlenmeli.
- `src/` klasörü henüz oluşturulmamış.
- Eğitim/değerlendirme pipeline'ı yok — sıradaki büyük iş.

## Bağlantılı Sayfalar

- [[synthesis/project-overview]]
- [[synthesis/roadmap]]
- [[concepts/data-cleaning-pipeline]]
- [[entities/oxford-flowers-102]]
