# HViT-GMP: Hybrid Vision Transformer for Autism Severity Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Thesis_Research-purple)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Overview
**Project Title:** *HVIT-GMP: Hybrid Vision Transformer with Global Masked Pooling for Autism Severity Estimation using Eye-Tracking Biomarkers* **Author:** Md. Hazrot Ali (KUET)  
**Supervisor:** Dr. Monira Islam  

This repository contains the official implementation of my undergraduate thesis. Unlike traditional "Binary" (ASD vs. Non-ASD) models, **HViT-GMP** is designed to estimate Autism Spectrum Disorder (ASD) severity across **four distinct levels** (High, Medium, Mild, Low).

To overcome the scarcity of medical eye-tracking data, this framework utilizes a **Two-Stage Self-Supervised Learning (SSL)** pipeline, training an "Anatomy Expert" (Masked Autoencoder) before fine-tuning for severity classification.

## 🔬 Key Innovations
* **Multi-Class Severity:** Moves beyond binary diagnosis to distinct clinical grades: *High, Medium, Mild, Low*.
* **Global Masked Pooling (GMP):** A novel aggregation layer that replaces standard Global Average Pooling. It acts as a hard-attention mechanism, prioritizing pupil/iris dynamics while masking out 85% of irrelevant skin/sclera noise.
* **Self-Supervised Pre-training:** Uses a Masked Autoencoder (MAE) in Phase 1 to learn robust anatomical features from unlabeled data, solving the overfitting problem common in small medical datasets.

## 🏗️ Architecture Pipeline
The model operates in two distinct phases:

### Phase 1: The "Anatomy Expert" (Self-Supervised)
* **Goal:** Learn structure without labels.
* **Method:** We mask **75%** of the input eye image. The model (EfficientNetV2 Backbone) must reconstruct the missing pixels.
* **Result:** The model learns "what an eye looks like" (pupil edges, iris texture) without needing diagnostic labels.

### Phase 2: Severity Classification (Supervised)
* **Goal:** Diagnose ASD Severity.
* **Method:** We freeze the backbone and attach a **Hybrid Vision Transformer (ViT)** head with **Global Masked Pooling**.
* **Output:** 4-Class Softmax Probability (High, Med, Mild, Low).

## 📊 Performance
Validated on a balanced dataset of **4,000 Eye Images**.

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | **90.50%** |
| **AUC (Normal Class)** | 0.97 |
| **AUC (Severe Class)** | 0.96 |
| **Inference Speed** | ~45ms / image |

> **Visual Evidence:** Grad-CAM heatmaps confirm the model focuses exclusively on the **Pupil-Iris boundary**, ignoring background noise (eyelashes/skin).

## 📂 Repository Structure
| File | Description |
| :--- | :--- |
| `Eye_tracking_autism_preprocessing.ipynb` | **Step 1:** Extracts 468 landmarks using MediaPipe, applies CLAHE enhancement, and performs pupil-centered cropping. |
| `SSL_first_phase.ipynb` | **Step 2:** Trains the Masked Autoencoder (MAE) to reconstruct masked eye images (Self-Supervised). |
| `Hybrid_model.ipynb` | **Step 3:** Loads Phase 1 weights and fine-tunes the Hybrid ViT + GMP for severity estimation. |


## 🛠️ Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/HazrotAli/HViT-GMP.git](https://github.com/HazrotAli/HViT-GMP.git)
    cd HViT-GMP
    ```
2.  Install dependencies:
    ```bash
    pip install tensorflow opencv-python mediapipe matplotlib scikit-learn
    ```

## 📝 Citation
If you find this code useful for your research, please cite this thesis:

```bibtex
@thesis{ali2026hvitgmp,
  title={HVIT-GMP: Hybrid Vision Transformer with Global Masked Pooling for Autism Severity Estimation},
  author={Ali, Md. Hazrot},
  school={Khulna University of Engineering & Technology (KUET)},
  year={2026},
  type={Bachelor's Thesis}
}
