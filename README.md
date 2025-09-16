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

### ***The model achieves a test accuracy of 98.17%.***

# **2. 🗂️ Dataset**

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

# **3. 🛠️ Methodology**

* ## **🔄Data Preprocessing & 🎛️Augmentation:**
To improve model generalization and prevent overfitting, extensive data augmentation was applied to the training dataset.

### **🧪 Training Transforms:**
* ***Resize(256):*** Resizes images to 256x256 pixels.
* ***RandomHorizontalFlip():*** Randomly flips images horizontally.
* ***RandomRotation(35):*** Randomly rotates images by up to 35 degrees.
* ***ColorJitter():*** Randomly changes the brightness, contrast, and saturation.
* ***CenterCrop(224):*** Crops the image to 224x224 from the center.
* ***ToTensor():*** Converts images to PyTorch tensors.
* ***ormalize():*** Normalizes tensors with ImageNet's mean and standard deviation.

### **🧪 Validation & Test Transforms:**
Only essential preprocessing steps (Resize, CenterCrop, ToTensor, Normalize) were applied to the validation and test sets to ensure a consistent evaluation.

* ## **💻 Model Architecture:**  
The core of this project is a custom model built upon the EfficientNet-B4 architecture, pre-trained on the ImageNet dataset. This approach utilizes transfer learning.
1. ***🧱 Base Model:***  
efficientnet_b4 is loaded from the timm (PyTorch Image Models) library.
2. ***✂️ Feature Extractor (Backbone):***  
The main convolutional blocks of EfficientNet-B4 are used as a feature extractor. The weights of these layers are frozen (requires_grad = False), so they are not updated during training.
3. ***➕ Custom Deconvolution Block (CustomDeconvCNN):***  
A custom module was designed to refine the features extracted by the backbone. This block consists of:
    * Two Transposed Convolution (ConvTranspose2d) layers to upsample the feature maps.
    * Two standard Convolution (Conv2d) layers for further feature processing.
This block is inserted between the backbone and the final classification head.
4. ***➕ Custom Classifier (CustomClassifier):***  
The original classifier is replaced with a custom fully connected head that includes:
    * A linear layer mapping input features to 512 hidden units.
    * A ReLU activation function.
    * A Dropout layer (p=0.3) to prevent overfitting.
    * A final linear layer mapping the 512 units to the 4 output classes.

### The final model (CustomEfficientNetB4) integrates these components in the following forward pass sequence:
### `Image -> Backbone -> CustomDeconvCNN -> Original ConvHead -> Global Pooling -> CustomClassifier -> Output`

* ## **🎯 Training Process:**
* ***Environment:*** The model was trained in a Google Colab environment using a T4 GPU.
* ***Hyperparameters:***
    * ***Optimizer: Adam with a learning rate of 0.001 and weight decay of 1e-4.***
    * ***Loss Function: CrossEntropyLoss, suitable for multi-class classification.***
    * **Epochs: 30.**
    * **Batch Size: 64.**

# **4. 📊 Results**
The model demonstrated excellent performance on the test set, validating its effectiveness in classifying grapevine diseases.

* ## ***Performance Metrics:***
    * ***Train Accuracy: 98.75%***
    * ***Test Accuracy: 98.17%***

* ## ***Confusion Matrix:***
**The confusion matrix highlights the model's high accuracy for each class, with very few misclassifications.**

* ## ***Classification Report:***
The classification report provides a detailed breakdown of the model's performance, showing high precision, recall, and F1-scores for all four classes.

**✅ Class-wise metrics**
| Class                                | Precision | Recall | F1-Score | Support |
|--------------------------------------|-----------|--------|----------|---------|
| Grape___Black_rot                    | 1.00      | 0.93   | 0.96     | 472     |
| Grape___Esca_(Black_Measles)         | 0.94      | 1.00   | 0.97     | 480     |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1.00 | 1.00   | 1.00     | 430     |
| Grape___healthy                      | 1.00      | 1.00   | 1.00     | 423     |

**✅ Summary metrics**
| Metric        |           |        |          |         |
|---------------|-----------|--------|----------|---------|
| **Accuracy**  |           |        | **0.98** | 1805    |
| **Macro avg** | 0.98      | 0.98   | 0.98     | 1805    |
| **Weighted avg** | 0.98   | 0.98   | 0.98     | 1805    |


# **5. 🚀 How to Replicate**

* ### **⚒️ Prerequisites:**  
Make sure you have Python 3 and the following libraries installed. You can install them using pip:  
```pip install torch torchvision numpy matplotlib scikit-learn timm```

* ### **⚙️ Setup**
**1. 🧬 Clone the Repository:**  
`git clone https://github.com/Anindya4/Plant-leaf-disease-recognition-2.git`  
`cd Plant-leaf-disease-recognition-2`

**2. Dataset:**  
Download the Grapevine Leaves Image Dataset from files and place the GrapeVine.zip file in a directory accessible by the notebook (e.g., `/content/drive/MyDrive/Dataset/ in Google Colab`).

**Update Paths:**  Modify the file paths in the notebook to match the location of your dataset.

**Run the Notebook:**  Execute the cells in the `Notebook.ipynb` file sequentially.

# **6. ⚠️ Note on Implementation**

An important detail was discovered during the analysis of the notebook. Although a `CustomEfficientNetB4` class with a deconvolutional block was defined, the training loop in `Cell 33` was executed on the original `timm` model instance (model), not the `custom one (custom_model)`.   

### `Due to the lack of resources as Colab provided resouces for very limited amount of time.`  
The excellent results presented are therefore for a standard fine-tuned EfficientNet-B4, where the optimizer trained the unfrozen final classifier.

### ***To train the intended custom architecture, the variable in the training loop should be changed from `model` to `custom_model`. This would be an interesting experiment to see how the custom deconvolution block affects performance.***