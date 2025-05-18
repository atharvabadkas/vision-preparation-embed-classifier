# 🍽️ Food Preparation Classification  
**Using OpenCLIP (ViT-L/14) and Few-Shot Feature Tuning**

---

## 📖 Overview

CLIPPrep is a feature-tuned food preparation image classification framework built on top of **OpenCLIP (ViT-L/14)** and a lightweight **linear SVM**. Designed to classify ingredients from prep station images, the system is optimized for **few-shot learning**, **robust augmentation**, and **real-world kitchen deployment**. It supports batch inference, test-time augmentation (TTA), top-k predictions, and confidence thresholds—making it ideal for food tech analytics and sustainability monitoring.

---

## 🚀 Features

- 🔎 Embedding generation via OpenCLIP (ViT-L/14)
- 🎯 Linear SVM trained with class weights and balanced sampling
- 🔁 Test-Time Augmentation with ensemble voting
- ⚖️ Class imbalance handling via oversampling and weighted training
- ✅ Confidence thresholds and top-3 predictions
- 🗂️ Batch prediction with robust error handling
- 📊 CSV-based logging and real-time progress monitoring
- 🧪 v1 vs v2 architecture with measurable performance uplift

---

## 📁 Code Files

| File | Purpose |
|------|---------|
| `verandah_prep_training.py` | v1: Basic embedding generation + SVM training |
| `varandah_prep_training.py` | ✅ v2: Balanced sampling, class weights, TTA |
| `verandah_prep_model.py` | v1 inference: Simple averaging |
| `varandah_prep_model.py` | ✅ v2 inference: Majority voting, thresholds |
| `Vision Model Experiments - Prep Data Feature Tuning (T1).csv` | Model tuning performance |
| `Vision Model Experiments - Prep Dataset - Django Testing (T2).csv` | Real-world test results |

---

## 🧪 v1 vs v2: Why v2 Performs Better

### ✅ v2 Improvements:
- **Balanced sampling** of classes with `samples_per_class`  
- **10 augmentations per image** (vs 5 in v1)  
- **Confidence threshold filtering** for rejecting low-confidence predictions  
- **Ensemble logic**: Combines majority voting + probability averaging  
- **Error handling**: Supports invalid formats, grayscale/RGBA, and NaNs  
- **Top-3 prediction output** with confidence scores  
- **Batch support**: Efficient, real-time folder processing  

### 🚫 v1 Limitations:
- Simple average across augmentations  
- No class balancing or class weights  
- No robustness checks or thresholding  
- No debugging/logging for data edge cases

---

## 🧠 Model Training Pipeline

### 🔹 Steps (v2):
1. Resize + augment image (10x per image)
2. Generate OpenCLIP ViT-L/14 embeddings
3. Apply **class-balanced sampling**
4. Train linear SVM with class weights
5. Save classifier + label encoder

## 🔍 Inference Pipeline (v2)

1. **Load** input image or folder
2. **Apply** test-time augmentations (e.g. flip, crop, jitter)
3. **Generate** OpenCLIP (ViT-L/14) embeddings
4. **Ensemble Prediction** using:
   - Majority Voting
   - Average Probability
   - Confidence Threshold Filtering
5. **Export** predictions to CSV + print console summary

### 🔹 Single Image Inference

python varandah_prep_model.py
  --input_image ./sample.jpg
  --classifier prep_classifier.pkl
  --encoder prep_encoder.pkl
  --n_aug 10
  --confidence 0.05

## 🧪 Test-Time Augmentation (TTA)

Augmentations:
- Horizontal Flip
- Random Crop & Resize
- Color Jitter
- Brightness/Contrast Adjustment
- Rotation (minor)

Prediction logic:
- Invalid augmentations are skipped
- Ensemble = majority vote + average probability
- Confidence threshold filters out weak predictions

---

## 📊 Results Summary

### 📈 `Vision Model Experiments - Prep Data Feature Tuning (T1).csv`

| Experiment                  | Accuracy (%) | Notes                          |
|-----------------------------|--------------|--------------------------------|
| v1 Basic Embeddings         | 84.3         | No class balancing             |
| ✅ v2 Balanced + Weighted    | **91.7**     | +7.4% improvement from v1      |
| v2 No Ensemble              | 88.5         | Dropped accuracy without voting |

> v2 demonstrates clear gains in performance through balancing, TTA, and ensemble logic.

## 🧪 Real-World Testing: `Prep Dataset - Django Testing (T2).csv`

| Scenario                     | Accuracy (%) |
|-----------------------------|--------------|
| Plate (Mixed Veggies)       | 90.2         |
| Prep Tray (Dry Masala)      | 91.6         |
| Bowl (Curry Ingredients)    | 89.7         |
| **Average Accuracy**        | **90.5**     |

> 🧪 In deployment tests with Django pipelines, CLIPPrep maintains >90% accuracy.

## 🧠 Design Goals

- 🔁 Few-shot generalization using CLIP embeddings
- ⚖️ Robust to class imbalance through sampling and weighting
- 📤 Scalable folder-based inference and batch support
- 🧪 Transparent predictions: top-3 labels + confidence
- 🔧 Flexible thresholding and augmentations for tuning performance
