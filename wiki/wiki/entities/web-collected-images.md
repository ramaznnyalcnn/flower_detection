---
title: Web'den Toplanan Görseller (Bing)
category: entity
summary: bing-image-downloader ile toplanmış, gerçek dünya çeşitliliğine sahip ikinci aşama eğitim verisi.
tags: [dataset, real-world, transfer]
sources: 1
updated: 2026-05-11
---

# Web'den Toplanan Görseller

## Nedir?

`download_google_images.py` (aslında bing-image-downloader kullanır) ile her sınıf için web'den çekilmiş çiçek görselleri. `data/raw/google_images/` altında saklanır.

## Projedeki Rolü

Gerçek dünya dayanıklılık testinin verisi. Telefonla çekilmiş, farklı arka planlı, farklı çözünürlüklerde, bazıları bulanık görseller içerir. [[concepts/data-cleaning-pipeline]] bu set için **kritik** — Oxford 102 zaten temiz.

## İlgili Sayfalar

- [[entities/oxford-flowers-102]]
- [[concepts/data-cleaning-pipeline]]
- [[sources/project-readme]]
