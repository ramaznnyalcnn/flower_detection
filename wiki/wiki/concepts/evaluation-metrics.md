---
title: Değerlendirme Metrikleri
category: concept
summary: Accuracy, precision, recall, F1, confusion matrix, ROC/AUC — multi-class fine-grained problem için.
tags: [metrics, evaluation, confusion-matrix, roc]
sources: 1
updated: 2026-05-11
---

# Değerlendirme Metrikleri

## Temel Metrikler

- **Accuracy** (top-1, top-5) — fine-grained problemlerde top-5 anlamlı
- **Precision / Recall / F1** — sınıf başına + macro/weighted average
- **Confusion matrix** — 102×102, hangi sınıflar karışıyor?
- **ROC eğrisi + AUC** — one-vs-rest, sınıf başına

## Karşılaştırma Tablosu (planlanan)

| Model | Top-1 Acc (Oxford) | Top-1 Acc (Web) | Eğitim süresi | Inference (ms) | Model boyutu |
|-------|---------------------|------------------|----------------|-----------------|---------------|
| SVM-RBF | TBD | TBD | TBD | TBD | TBD |
| RF | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| ResNet50 | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B0 | TBD | TBD | TBD | TBD | TBD |

## Dayanıklılık Testi

Asıl ilginç soru: Oxford'da %X başarılı bir model, web görsellerinde ne kadar düşüyor? **Generalization gap** = `acc(oxford) - acc(web)`. Bu sayı projenin ana çıktısı.

## Bağlı Sayfalar

- [[concepts/classical-classifiers]]
- [[concepts/deep-learning-approach]]
- [[synthesis/project-overview]]
