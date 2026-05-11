---
title: Derin Öğrenme Yaklaşımı (CNN, Transfer Learning)
category: concept
summary: ResNet/EfficientNet üzerinde transfer learning, CUDA hızlandırma, fine-tuning stratejisi.
tags: [cnn, transfer-learning, resnet, efficientnet, pytorch, cuda]
sources: 1
updated: 2026-05-11
---

# Derin Öğrenme Yaklaşımı

## Yaklaşım

ImageNet üzerinde önceden eğitilmiş bir omurga (ResNet50 veya EfficientNet-B0) alıp 102 sınıfa **transfer learning** uygulamak. PyTorch + torchvision, CUDA hedefli.

## İki Aşamalı Fine-Tuning

1. **Aşama 1 — Sadece classifier head:** Omurga donduruldu, son FC katmanı eğitildi. [[entities/oxford-flowers-102]] üzerinde.
2. **Aşama 2 — Son blok serbest:** Son convolutional blok + head birlikte fine-tuned. [[entities/web-collected-images]] dahil edilir.

## Tipik Hiperparametreler

- Optimizer: AdamW (lr=1e-4 head, 1e-5 omurga)
- Scheduler: CosineAnnealing
- Augmentation: random crop, flip, color jitter, random erasing
- Batch size: GPU belleğine göre (32-64)
- Epoch: erken durdurma (val loss patience=5)

## Karşılaştırma Hedefi

[[concepts/classical-classifiers]] ile aynı test setinde. Hipotez: CNN doğruluk + genellemede klasiklerin üzerinde olur ama eğitim maliyeti yüksek.

## Bağlı Sayfalar

- [[concepts/classical-classifiers]]
- [[concepts/evaluation-metrics]]
- [[entities/oxford-flowers-102]]
