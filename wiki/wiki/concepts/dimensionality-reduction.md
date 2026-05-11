---
title: Boyut İndirgeme
category: concept
summary: PCA, LDA, t-SNE, UMAP — yüksek boyutlu feature uzayını sıkıştırma ve görselleştirme.
tags: [pca, lda, tsne, umap, visualization]
sources: 1
updated: 2026-05-11
---

# Boyut İndirgeme

## Yöntemler ve Rolleri

| Yöntem | Tip | Kullanım |
|--------|-----|----------|
| **PCA** | Lineer, denetimsiz | Varyans odaklı sıkıştırma, gürültü azaltma |
| **LDA** | Lineer, denetimli | Sınıflar arası ayrışmayı maksimize — klasifier öncesi |
| **t-SNE** | Non-lineer, denetimsiz | Görselleştirme (2D), küme yapısını incelemek |
| **UMAP** | Non-lineer, denetimsiz | Görselleştirme + bazen feature input — t-SNE'den hızlı |

## Pipeline'daki Yeri

[[concepts/feature-extraction]] çıkışı yüksek boyutlu (tipik 1000-5000). PCA/LDA → [[concepts/classical-classifiers]] girişi. t-SNE/UMAP daha çok rapor görselleri için.

## Açık Sorular

- LDA'nın 102 sınıflı problemde nereye kadar yardımı olacağı (max bileşen = n_classes-1 = 101)
- PCA varyans eşiği: %95 mi %99 mu?

## Bağlı Sayfalar

- [[concepts/feature-extraction]]
- [[concepts/classical-classifiers]]
