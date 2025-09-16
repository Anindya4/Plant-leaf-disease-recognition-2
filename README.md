# **Plant leaf disease recognition (using Grapevine dataset)**
This project demonstrates the use of a deep learning model to classify plant leaf diseases using Grapvine dataset downloaded from Kaggle. The dataset contains 4 classes in which 3 are diseased class one is healthy. The model leverages a pre-trained EfficientNet-B4 architecture, customized with additional layers, and is built using PyTorch.

# **📋 Table of Contents**

1. Overview
2. Dataset
3. Methodology
    - Data Preprocessing & Augmentation
    - Model Architecture
    - Training Process
4. Results
    - Performance Metrics
    - Training & Validation Curves
    - Confusion Matrix
    - Classification Report
6. How to Replicate
    - Prerequisites
    - Setup
7. Note on Implementation

# **1. 📖 Overview**

The goal of this project is to build an accurate and robust image classification model for identifying diseases in grapevine leaves. Early and accurate detection of diseases like Black Rot, Esca, and Leaf Blight is crucial for maintaining crop health and yield. This notebook covers the entire workflow from data loading and augmentation to model building, training, and evaluation.

***The model achieves a test accuracy of 98.17%.***

# **2. 🖼️ Dataset**

The model is trained on a dataset of grapevine leaf images, which are organized into four classes:
* Grape___Black_rot
* Grape___Esca_(Black_Measles)
* Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
* Grape___healthy

The dataset is split as follows:
* Initial Training Set: 7,222 images
* Test Set: 1,805 images

The initial training set was further divided into a final training set (85%) and a validation set (15%) using stratified sampling to ensure that the class distribution remained consistent across both splits.
* Final Training Set: 6,138 images
* Validation Set: 1,084 images
