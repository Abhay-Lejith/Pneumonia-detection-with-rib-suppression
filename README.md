# Chest X-Ray Analysis: Rib Suppression & Pneumonia Detection

This repository contains deep learning implementations for processing and analyzing chest X-ray (CXR) images. It addresses two critical tasks in medical image analysis: removing bony structures (rib suppression) to clear the lung field, classifying images to detect pneumonia using transfer learning and also using GradCAM to visualize key areas of the x-rays for decision making.

## Datasets

### 1. Rib Suppression
https://www.kaggle.com/datasets/hmchuong/xray-bone-shadow-supression

### 2. Pneumonia Detection
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia


## Repository Contents

### 1. `rib_suppression_model.ipynb`
A preprocessing model designed to isolate and suppress rib structures in chest X-rays, allowing for a clearer view of the underlying soft tissue (lungs, heart, etc.).

* **Architecture:** U-Net (Encoder-Decoder with skip connections).
* **Input Resolution:** 256x256 grayscale images.
* **Loss Function:** Mean Squared Error (MSE).


### 2. `pneumonia_detection.ipynb`
A binary classification model to diagnose Pneumonia from chest X-ray images using Transfer Learning.

* **Architecture:** VGG16 (pre-trained on ImageNet) with custom fully connected heads.
* **Input Resolution:** 150x150 RGB images.
* **Augmentation:** Includes rotation, shear, zoom, and horizontal flipping to prevent overfitting.
* **Training Strategy:**

    1.  **Fine-Tuning:** The last 4 convolutional layers of VGG16 are unfrozen and retrained with a very low learning rate (1e-5) to adapt the feature extractors to medical X-rays.
