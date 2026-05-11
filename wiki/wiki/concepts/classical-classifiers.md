---
title: Klasik Sınıflandırıcılar
category: concept
summary: SVM (linear/RBF), k-NN, Naive Bayes, Random Forest, XGBoost — el yapımı özellikler üzerinde çalışan modeller.
tags: [svm, knn, random-forest, xgboost, naive-bayes, classical-ml]
sources: 1
updated: 2026-05-11
---

# Klasik Sınıflandırıcılar

[[concepts/feature-extraction]] çıktısı üzerinde eğitilen sınıflandırıcılar.

## Kapsam

| Model | Kernel/Hiperparametre | Beklenti |
|-------|------------------------|-----------|
| SVM (Linear) | C grid | Hızlı baseline |
| SVM (RBF) | C, γ grid | En güçlü klasik aday |
| k-NN | k ∈ {1,3,5,7}, ağırlıklı/uniform | Basit referans |
| Naive Bayes | Gaussian | Hız referansı, düşük doğruluk beklenir |
| Random Forest | n_estimators, max_depth | Robust, interpretable |
| XGBoost | gradient boosted trees | En güçlü tree-based aday |

## Eğitim Stratejisi

- Stratified K-Fold (k=5)
- Grid search ile hyperparam tuning
- Çıktı: en iyi model + cross-validation skoru
- Sonuçlar [[concepts/evaluation-metrics]] üzerinden raporlanır

## Karşılaştırma Hedefi

[[concepts/deep-learning-approach]] ile aynı test setinde karşılaştırılır — doğruluk, hız, model büyüklüğü, gerçek dünya görsellerine genelleme.

## Bağlı Sayfalar

- [[concepts/feature-extraction]]
- [[concepts/deep-learning-approach]]
- [[concepts/evaluation-metrics]]
