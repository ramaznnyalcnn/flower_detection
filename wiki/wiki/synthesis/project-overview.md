---
title: Proje Genel Bakış
category: synthesis
summary: Çiçek tanıma projesinin yüksek seviyeli özeti — amaç, yaklaşım, beklenen çıktılar.
tags: [overview, summary, thesis]
sources: 1
updated: 2026-05-11
---

# Proje Genel Bakış

## Tek Cümleyle

Klasik örüntü tanıma yöntemleriyle CNN'i, **gerçek dünya görsellerinde dayanıklılık** ekseninde karşılaştıran bir çiçek sınıflandırma çalışması.

## Tez

> "Temiz veriyle eğitilmiş bir model, telefonla çekilmiş rastgele bir çiçek fotoğrafını tanıyabilir mi?"

İki aşamalı eğitim stratejisi bu sorunun cevabını ölçer:
1. [[entities/oxford-flowers-102]] — temiz baseline
2. [[entities/web-collected-images]] — gerçek dünya

## Karşılaştırma Eksenleri

| Eksen | Klasik | Derin Öğrenme |
|-------|--------|---------------|
| Özellik | El yapımı ([[concepts/feature-extraction]]) | Öğrenilmiş |
| Veri ihtiyacı | Düşük | Yüksek (transfer ile azalır) |
| Inference | Hızlı (CPU yeterli) | Yavaş (GPU faydalı) |
| Yorumlanabilirlik | Yüksek | Düşük |
| Beklenen accuracy | Orta | Yüksek |
| Beklenen genelleme | Düşük | Yüksek |

## Çıktılar

1. **Karşılaştırma tablosu** — [[concepts/evaluation-metrics]] sayfasındaki şablon doldurulmuş hâliyle.
2. **Confusion matrix'leri** — hangi çiçek türleri karışıyor?
3. **Boyut indirgeme görselleştirmeleri** — [[concepts/dimensionality-reduction]] (t-SNE/UMAP).
4. **Streamlit/Gradio arayüzü** — kullanıcı kendi fotoğrafını yükleyip test edebilir.
5. **Rapor/sunum** — pattern recognition dersi için.

## Yol Haritası

Detaylı plan: [[synthesis/roadmap]].

## Bağlı Sayfalar

- [[synthesis/roadmap]]
- [[sources/project-readme]]
- [[concepts/classical-classifiers]]
- [[concepts/deep-learning-approach]]
- [[concepts/evaluation-metrics]]
