# Geliştirme Notları

## Sınıf Stratejisi (Faz A revizyonu)

**Karar:** 104 sınıf → ~20-25 sınıf, sınıf başına 300+ görüntü.

**Neden:** Az veriyle 104 sınıf öğretmek demo'da hata üretir. Az sınıf + çok veri = yüksek doğruluk + web görsellerinde tutarlılık.

**Yapılacak:**
- `configs/selected_classes.yaml` — tutulacak sınıf listesi
- `build_unified_dataset.py` — `--classes` filtresi
- Faz C web scraping sadece seçili sınıflar için

**Hedef sınıflar (~20):**
rose, tulip, sunflower, daisy, iris, orchid, carnation, lily, peony,
lavender, hydrangea, daffodil, chrysanthemum, poppy, hibiscus,
camellia, bougainvillea, lotus, freesia, marigold
