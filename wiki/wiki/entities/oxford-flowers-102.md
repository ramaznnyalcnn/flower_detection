---
title: Oxford Flowers 102
category: entity
summary: Visual Geometry Group (Oxford) tarafından yayınlanan 102 sınıflı, ~8189 görselli çiçek veri seti.
tags: [dataset, benchmark, baseline-data]
sources: 1
updated: 2026-05-11
---

# Oxford Flowers 102

## Nedir?

Oxford VGG grubunun 2008'de yayınladığı, İngiltere'de yaygın 102 çiçek türünü içeren sınıflandırma veri seti. Her sınıfta 40-258 arası görsel var; toplamda ~8189 görsel. Standart bir benchmark olarak fine-grained classification literatüründe sıkça kullanılır.

## Projedeki Rolü

İlk aşama eğitim verisi. **Temiz, kontrollü** koşullarda — modelin tanıma kapasitesini ölçmek için kullanılır. Sonraki aşamada [[entities/web-collected-images]] ile yapılan ikinci eğitim, bu temizlikten gerçek dünyaya geçişi test eder.

## Erişim

`download_oxford102.py` script'i ile indirilir. Çıktı: `data/raw/oxford102/`.

## İlgili Sayfalar

- [[sources/project-readme]]
- [[synthesis/project-overview]]
