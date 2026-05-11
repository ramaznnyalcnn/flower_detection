---
title: Özellik Çıkarımı (Feature Extraction)
category: concept
summary: HOG, LBP, GLCM, Gabor, HSV histogram, Hu moments — klasik pipeline'ın girdisi.
tags: [features, hog, lbp, glcm, color, shape]
sources: 1
updated: 2026-05-11
---

# Özellik Çıkarımı

Klasik makine öğrenmesi pipeline'ında [[concepts/data-cleaning-pipeline]] çıktısından elle tasarlanmış özellikler üretilir.

## Özellik Grupları

### Renk
- HSV histogram (uzay: HSV — aydınlatma değişimine HSV daha dayanıklı)
- Renk momentleri (ortalama, std, skewness her kanal için)
- Dominant renk (k-means)

### Doku
- **LBP** (Local Binary Patterns) — yerel doku örüntüleri
- **GLCM** (Gray-Level Co-occurrence Matrix) — contrast, dissimilarity, homogeneity, energy, correlation
- **Gabor filtre** bankası — yönlü doku tepkileri

### Şekil
- **HOG** (Histogram of Oriented Gradients) — kenar yönü dağılımı
- **Hu Moments** — invariant şekil tanımlayıcıları

## Çıktı

Her görsel için bir feature vector (boyut tasarıma göre değişir, tipik 1000-5000). [[concepts/classical-classifiers]] girdisi olur. Yüksek boyutlu olduğunda [[concepts/dimensionality-reduction]] devreye girer.

## Açık Sorular

- Hangi özellik grubu en çok ayırt edici? (LDA + tek tek değerlendirme)
- Feature normalization stratejisi: StandardScaler mı, MinMax mı?

## Bağlı Sayfalar

- [[concepts/classical-classifiers]]
- [[concepts/dimensionality-reduction]]
- [[sources/project-readme]]
