---
title: Yol Haritası
category: synthesis
summary: Mevcut durumdan bitmiş projeye doğru sıralanmış iş paketleri, bağımlılıklar ve karar noktaları.
tags: [roadmap, planning, milestones]
sources: 1
updated: 2026-05-11
---

# Yol Haritası

> Mevcut durum (2026-05-11): Veri indirme + temizleme hazır. `src/` klasörü henüz yok. Bir sonraki büyük iş **feature extraction + ilk klasik baseline**.

## Faz 0 — Temizlik (yarım gün)

- [ ] Projedeki `{data` adlı garip dosya/klasörü incele ve kaldır.
- [ ] `src/` klasörünü oluştur, `__init__.py` ekle.
- [ ] `data/`, `models/`, `results/` klasörlerinin var olduğundan emin ol (gitignore'a ekle).
- [ ] (Opsiyonel) Git başlat — şu an git repo değil, versiyonlama için faydalı olur.

## Faz 1 — Veri Hazır mı? Doğrula (yarım gün)

- [ ] `download_oxford102.py` çalıştır, sınıf başına görsel sayısını doğrula.
- [ ] `download_google_images.py --limit 50` çalıştır, en az 10-20 sınıf için web verisi indir.
- [ ] `clean_data.py --source all` çalıştır, `data/processed/` çıktısını ve sınıf dengesizliği raporunu incele.
- **Karar noktası:** Dengesizlik fazla mı? Az verili sınıflar için augmentation mı, exclude mu?

## Faz 2 — Klasik Pipeline Baseline (2-3 gün)

İlk hedef: **çalışan tek bir end-to-end klasik pipeline**, kalite optimizasyonu sonra.

- [ ] `src/feature_extraction.py` — başlangıçta sadece **HSV histogram + HOG** (en yüksek getirili 2 özellik).
- [ ] `src/classifiers.py` — sadece **SVM-RBF + Random Forest** ile başla.
- [ ] `src/evaluation.py` — accuracy, confusion matrix, classification report.
- [ ] `run.py` — `--task classical-baseline` ile bu pipeline'ı çalıştırır, sonucu `results/` altına yazar.
- **Çıktı:** Oxford 102 üzerinde ilk doğruluk skoru. Hedef: >%50 (rastgele = %1).

## Faz 3 — Klasik Pipeline Genişletme (2 gün)

- [ ] LBP, GLCM, Gabor, color moments, Hu moments ekle ([[concepts/feature-extraction]] kapsamı).
- [ ] k-NN, Naive Bayes, XGBoost ekle ([[concepts/classical-classifiers]]).
- [ ] Hyperparameter tuning (GridSearchCV).
- [ ] [[concepts/dimensionality-reduction]] — PCA + LDA boyut indirgeme entegre et.
- **Karar noktası:** Hangi feature kombinasyonu en iyi? Ablation tablosu üret.

## Faz 4 — Derin Öğrenme Pipeline (2-3 gün)

- [ ] `src/cnn_model.py` — ResNet50 transfer learning, sadece head eğitimi.
- [ ] Augmentation pipeline (torchvision transforms).
- [ ] Eğitim döngüsü + checkpoint + early stopping.
- [ ] Aşama 2 fine-tuning (son blok serbest).
- **Çıktı:** Oxford 102 üzerinde CNN doğruluk skoru, sınıf başına metrikler.

## Faz 5 — İkinci Aşama Eğitim ve Dayanıklılık Testi (1-2 gün)

- [ ] Web görselleriyle ikinci aşama eğitim — hem klasik hem CNN için.
- [ ] **Generalization gap** ölç: `acc(oxford-test) - acc(web-test)`.
- [ ] [[concepts/evaluation-metrics]] tablosunu doldur.
- [ ] t-SNE/UMAP ile feature uzayını görselleştir.

## Faz 6 — Arayüz ve Rapor (1-2 gün)

- [ ] `app/app.py` — Streamlit ile drag-drop görsel yükleme + model seçimi + tahmin.
- [ ] En iyi modelleri `models/` altına kaydet.
- [ ] Tek bir özet rapor: `results/REPORT.md` veya PDF.
- [ ] README'yi son durumla güncelle.

## Faz 7 — Genişletme (opsiyonel)

- [ ] Ensemble (klasik + CNN voting/stacking).
- [ ] Grad-CAM ile CNN'in nereye baktığını görselleştir.
- [ ] Adversarial robustness mini-deneyi.

## Kritik Karar Noktaları

1. **Faz 1 sonu:** Web verisi yeterli mi? Eğer çok az/kalitesiz ise stratejiyi ayarla.
2. **Faz 2 sonu:** Baseline çok düşükse (< %30), veri ya da feature pipeline'ında sorun var demektir.
3. **Faz 4 sonu:** CNN klasiği yenmediyse hyperparam veya augmentation'da iyileştirme şart.
4. **Faz 5 sonu:** Generalization gap > %30 ise — web verisini artırmak veya domain adaptation eklemek gerekir.

## Bağlı Sayfalar

- [[synthesis/project-overview]]
- [[sources/project-readme]]
