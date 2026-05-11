---
title: Veri Temizleme Pipeline'ı
category: concept
summary: clean_data.py — boyut normalizasyonu, bulanıklık tespiti, duplicate eleme, renk uzayı ve format standardizasyonu.
tags: [preprocessing, pipeline, data-quality]
sources: 1
updated: 2026-05-11
---

# Veri Temizleme Pipeline'ı

## Amaç

Hem [[entities/oxford-flowers-102]] hem [[entities/web-collected-images]] kaynaklarından gelen görselleri tek tip, eğitime hazır bir formata indirger.

## Adımlar (clean_data.py içinde uygulanır)

1. **Boyut normalizasyonu:** Hedef 224×224 (standart CNN input), resize + padding.
2. **Bulanıklık tespiti:** Laplacian variance < 50 → elenmesi gereken görsel.
3. **Duplicate tespiti:** Perceptual hash (imagehash). OpenCV/imagehash yoksa MD5 fallback.
4. **Renk uzayı kontrolü:** Grayscale → RGB dönüşümü.
5. **Format standardizasyonu:** Hepsi JPG.
6. **Sınıf dengesizliği raporu:** Her sınıftaki görsel sayısı.

## Eşikler

- `MIN_ORIGINAL_SIZE = 64` px
- `BLUR_THRESHOLD = 50` (Laplacian variance)
- `MIN_FILE_SIZE = 3 KB`, `MAX_FILE_SIZE = 20 MB`

## Çıktı

`data/processed/` altına temizlenmiş görseller + bir JSON özet raporu.

## Bağlı Sayfalar

- [[sources/project-readme]]
- [[concepts/feature-extraction]] (çıktısı bunun girdisi)
