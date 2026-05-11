# Çiçek Tanıma Projesi — Uygulama Planı

Hocaya sunmalık, klasik öğrenci projesinin üstüne çıkacak gerçek dünyada çalışabilen çiçek tanıma sistemi.

---

## Hedef Mimari

**Tek görsel sınıflandırıcı yerine 3 katmanlı sistem:**

1. **Detector** (YOLOv8/RT-DETR) — buketteki her çiçeği bul (bbox)
2. **Classifier** (ResNet50/EfficientNet, çoklu-dataset üzerinde) — her crop'u sınıflandır
3. **Domain adapter** — laboratuvar → gerçek dünya boşluğunu kapatan fine-tune katmanı

Bu mimari klasik öğrenci projesini ayırır: "tek çiçek tanırım" değil, "buketteki 5 çiçeği say ve tanı" diyebilirsin.

---

## Faz A — Veri Birleştirme

**Çıktı:** `data/processed/unified/{train,val,test}/<sınıf_adı>/*.jpg`

- `scripts/build_unified_dataset.py` yaz:
  - Oxford 102 → 102 sınıf
  - TF Flowers (`tensorflow_datasets`) → 5 sınıf (daisy, dandelion, roses, sunflowers, tulips)
  - Kaggle "Flowers Recognition" (Alxmamaev) → 5 sınıf (chamomile, tulip, rose, sunflower, dandelion)
- **Ortak sınıf eşleştirme tablosu:** `configs/class_aliases.yaml` — `rose == roses == pink rose` gibi sinonimleri tek isme bağla
- Birleşim: ~102 sınıf, ortak olanlarda kaynak çeşitliliği artar
- Sınıf başına min/max görsel raporu: `results/unified_class_stats.json`
- 70/15/15 stratified split (`scripts/prepare_stratified_split.py` zaten var, parametre değiştir)

**Hocaya değer:** Sınıf çeşitliliği + kaynak karışımı dökümante edilir.

---

## Faz B — Güçlü Backbone Eğitimi

**Çıktı:** `models/unified_resnet50.pt` + `models/unified_efficientnet_b0.pt`

- `src/cnn_model.py`'i refactor et: backbone parametreli (`resnet50` / `efficientnet_b0` / `convnext_tiny`)
- 2 aşamalı eğitim:
  - Önce head freeze (5 epoch)
  - Sonra son 2 blok serbest (10 epoch)
- Augmentation: RandAugment + Mixup + RandomErasing (gerçek dünya için kritik)
- TTA (test-time augmentation) inference

**Hocaya değer:** Backbone karşılaştırma tablosu (params, FLOPs, accuracy, latency).

---

## Faz C — Web Veri Toplama

**Çıktı:** `data/raw/web/<sınıf_adı>/*.jpg`

- `download_google_images.py` var ama gözden geçir: Bing/DuckDuckGo da ekle (Google rate-limit)
- Otomatik etiket: klasör adı = sınıf adı
- **Otomatik temizlik:**
  - Perceptual hash ile duplicate sil
  - CLIP zero-shot ile "bu gerçekten X çiçeği mi?" filtresi (kötü etiketleri ele)
  - Min çözünürlük + blur threshold
- Hedef: sınıf başına 50-100 temiz görsel × en az 20 sınıf

**Hocaya değer:** "Etiketsiz veriyi nasıl etiketlettim" → CLIP filtreleme bir teknik katkı.

---

## Faz D — Domain Gap Ölçümü

**Çıktı:** `results/domain_gap_report.md` + grafikler

- Faz B modelini **dokunmadan** web seti üzerinde test et
- 4 metrik tablosu: `acc(oxford-test)`, `acc(unified-test)`, `acc(web-test)`, **gap**
- Sınıf bazlı gap: hangi sınıflar laboratuvardan gerçek dünyaya en çok düşüyor?
- t-SNE: oxford features vs web features — kümeler ayrışıyor mu? (domain shift kanıtı)

**Hocaya değer:** "Gerçek dünyada model neden çöker" sorusunu sayısal cevaplar.

---

## Faz E — İki Aşamalı Fine-Tune

**Çıktı:** `models/web_finetuned.pt`

- Web setini 50/50 böl: yarısı fine-tune'a, yarısı held-out test'e
- Düşük LR (1e-4), 5 epoch, sadece son blok + head
- **Catastrophic forgetting kontrolü:** fine-tune sonrası oxford-test'te ne kadar düşüyoruz?
- **Kıyas tablosu:**

  | Model | oxford-test | web held-out | gap |
  |---|---|---|---|
  | Pre-finetune | A | B | A-B |
  | Post-finetune | A' | B' | A'-B' |

**Hocaya değer:** Fine-tune'un faydası ve maliyeti sayısal.

---

## Faz F — Bouquet Detection (Projeyi Farklılaştıran Kısım)

**Çıktı:** `models/yolo_flower_detector.pt` + entegre inference

- COCO veya iNaturalist'ten "flower" bbox annotation'ları → veya Roboflow'da hazır flower-detection dataset (Open Images "flower" subset)
- YOLOv8n eğit: tek sınıf = "flower" (bbox tespiti)
- **Pipeline:**
  1. Görsel gelir → YOLOv8 bbox'ları çıkarır
  2. Her bbox crop'lanır → ResNet classifier sınıfı verir
  3. JSON çıktı: `[{bbox, class, confidence}, ...]`

**Hocaya değer:** "Buket içinde 3 gül, 2 papatya var" diyebilen tek öğrenci sensin.

---

## Faz G — Streamlit Demo + Rapor

- `app/app.py` güncelle: bbox overlay, sınıf etiketleri, confidence bar
- Hata analizi galeri: en kötü tahmin edilen 20 görsel + neden
- Grad-CAM ile "model nereye bakıyor" görselleştirmesi (3-4 örnek)
- `results/REPORT.md` final: bütün metrik tabloları, gap grafiği, ablation
- Sunum slaytları için key görseller

---

## Sunum / Rapor için "Wow" Noktaları

1. **Üç dataset birleştirme + alias tablosu** (veri mühendisliği gösterir)
2. **CLIP ile otomatik web-etiket temizleme** (modern teknik kullanımı)
3. **Domain gap'i sayısal ölçüm** (sadece accuracy değil, generalization)
4. **Catastrophic forgetting kontrolü** (literatürde fine-tune'da kimsenin bakmadığı şey)
5. **Bouquet detection** (sınıflandırma → tespit, bir seviye yukarı)
6. **Grad-CAM** (yorumlanabilirlik)

---

## Donanım

- GPU: RTX 3050 4GB yeterli (batch 16, mixed precision)
- Disk: TF Flowers + Kaggle ~1-2 GB; web veri ~500 MB

---

## Karar Noktaları

- **Faz A sonu:** Sınıf dengesizliği fazlaysa → augmentation ağırlığı veya az verili sınıfları çıkar.
- **Faz B sonu:** Unified-test'te <%60 ise backbone veya augmentation'da problem var.
- **Faz D sonu:** Gap >%30 ise Faz E zorunlu; <%10 ise Faz E opsiyonel, doğrudan Faz F'ye geç.
- **Faz E sonu:** Catastrophic forgetting >%15 ise EWC veya replay buffer ekle.
- **Faz F sonu:** Detector mAP <0.5 ise YOLOv8s'e geç veya daha fazla annotation topla.
