# Deep-Learning-In-Computer-Vision

## 🧠 Deep Learning with Python — Code Exercises (Computer Vision)

This repository contains my hands-on exploration of the **Computer Vision** chapters from the book [_Deep Learning with Python_ by François Chollet] (https://www.manning.com/books/deep-learning-with-python)(Creator of Keras). It includes annotated and structured implementations of the models presented in the book, with a focus on key concepts and practical applications.

---

## ⚠️ Disclaimer
The core code is adapted from the original book and is **not my own invention**. The goal is educational — to reproduce, understand, and build upon the author's work.

---

## 📘 Chapter 1 – Building a Simple ConvNet from Scratch (MNIST)

In this notebook, I implemented a **basic Convolutional Neural Network (CNN)** using Keras' Functional API, applied to the classic **MNIST handwritten digit classification task**.

### 🔍 What This Covers

- Loading and preprocessing grayscale image data from the MNIST dataset (28x28 pixels, single channel).
- Building a CNN with the following architecture:
  - Stacked **Conv2D** + **MaxPooling2D** layers
  - **Flatten** layer to transition from 3D to 1D tensor
  - Final **Dense** layer with **softmax** activation for classification
- Model compilation and training with `sparse_categorical_crossentropy` loss.

### 🧠 Key Deep Learning Concepts

- **Convolutional Layers**: Learn spatial hierarchies in image data using filters.
- **Max Pooling**: Downsamples feature maps by selecting maximum values (2×2 with stride 2) to reduce spatial dimensions.
- **Flattening**: Converts 3D feature maps into 1D vectors to connect to dense layers.
- **Padding & Stride**:
  - Padding helps maintain input size after convolution.
  - Stride affects how much the kernel moves, influencing output size.
  ---
## 📘 Chapter 2 – Image Classification & Data Augmentation
In this notebook, I implemented a **basic Convolutional Neural Network (CNN)** for a **Binary Image Classification Task**, applied to the **Dogs vs. Cats Kaggle Dataset** (https://www.kaggle.com/competitions/dogs-vs-cats). After the first training, in order to **mitigate overfitting** I applied **Data Augmentation to the training set** and **Dropout after the flatten layer**. With these techniques **overfitting occur much later during training.**

### 🔍 What This Covers
- Loading and preprocessing 5.000 RGB images from Dog vs. Cat Dataset (2.500 Dogs and 2.500 Cats) with shape (180,180,3)
- Added **Data Augmentation (horizontal flip, slight rotation, zoom)**
- Building a CNN with the following architecture:
  - **5 Conv2D** + **MaxPooling2D** layers
  - **Flatten** layer & **Dropout** Layer (0.5)
  - Final **Dense** layer with **sigmoid** activation for binary classification
- Model compilation and training with `binary_crossentropy` loss.
- By adding **Data Augmentation and Droput**, the **Test accuracy goes from 72% to 81%.**
### 🧠 Key Deep Learning Concepts
- **Data Augmentation**: generate more training data images by applying transformations to existing samples.
- **Droput Vs Data Augmentation**: In computer vision, the most used technique to mitigate overfitting is by the Data Augmentation the training set. That because Dropout after a Conv2D Layer may create disturbance. However, Droput can be still used in CNN models, but after the flatten layer.
---
## 📘 Chapter 3 – Trasfer Learning (VGG16 Model)
When working with a small dataset, using pretrained models is an extremely effective strategy.  
In this notebook, I will explore how to leverage pretrained convolutional networks (specifically **VGG16**) to classify images for the same previous task (cats vs dogs), using two main approaches: **Feature Extraction (Transfer Learning)** and **Fine-Tuning**.  
You’ll learn how to freeze model weights, preprocess image data, and build a custom classifier on top of learned features.

### 🔍 What This Covers
- Import and use pre-trained model
- (1) **Feature Extraction** with a pre-trained model
- (1) **Training a custom NN classifier** from scratch with **pre-trained model's prediction**.
- (2) **Freeze** pre-trained model weights.
- (2) **Remove top layers** of pre-trained model and **Add our custom layers.**
- (2) **Data Augmentation** on dataset training.
- (1-2) With these techniques the test accuracy **will reach 97% and 98%.**
### 🧠 Key Deep Learning Concepts
- **Pretrained Models**: When working with small datasets, it's often more effective to use a pretrained model rather than training one from scratch.
- **Feature Extraction (Transfer Learning)**: One common approach is to use **feature extraction**, (1) involves using the pretrained model to make predictions on the training data, and then using these predictions as the new training set for a custom model. (2) involves freezing the weights of a pretrained model and using it as the foundation for a custom model.
- **Data Augmentation in Transfer Learning**: To apply data augmentation in this context, you must use the second approach—freezing the pretrained model and attaching a new classifier—so that the augmented images are processed end-to-end.
---
## 📘 Chapter 4 – Fine-Tuning (VGG16 Model)
In this notebook, I will explore Fine-Tuning Technique on VGG16 Model. Fine-tuning is another popular technique that involves utilizing a pretrained model, and it complements feature extraction. The model will be used to classify images for the same previous task (cats vs dogs).

### 🔍 What This Covers
- **Add our custom model** to the top of pre-trained model
- **Freeze pre-trained model's weights**
- **Train the custom model**.
- **Unfreeze** some pre-trained model's weights.
- **Train both** unfreeze layers and part we added.
  
### 🧠 Key Deep Learning Concepts
- **Fine-Tuning**: is a method where we "unfreeze" the top layers of a pretrained convolutional base and train them along with the new classifier added to the model.
- **Freezing and Unfreezing Layers**: When fine-tuning a model, you start by freezing the initial layers of the pretrained network and then unfreeze the top layers to train them. Typically, you unfreeze only the final few layers because they capture more task-specific features, while the earlier layers capture more generic features like edges or textures.
- **Overfitting Considerations**: The more layers you unfreeze, the more parameters need to be trained, which increases the risk of overfitting.
---
## 📘 Chapter 5 – Image Segmentation (VGG16 Model)
In this notebook, I will explote **Image Segmentation task in Computer vision**. Specifically, I will focus on **semantic segmentation task** considering the **Oxford-IIIT Pets dataset**, which contains 7,390 images of dogs and cats, along with segmentation masks.

### 🔍 What This Covers
- Loading and preprocessing images and segmentation masks in 2 NumPy arrays
- Building a CNN with the following architecture:
  - Stacked **Conv2D** + **Conv2D Transpose** layers
  - Last **Conv2D Layer with 3 Filters** (num_classes) and **softmax function activation**.
- Compile the model with **sparse_categorical_crossentropy** and **sparse_categorical_accuracy**

### 🧠 Key Deep Learning Concepts
- **Semantic Segmentation**: A task in computer vision where each pixel in an image is classified into a category.
- **Conv2D and Conv2DTranspose Layers**: These are convolutional layers used in the model. Conv2D is used for downsampling the input, while Conv2DTranspose is used for upsampling and reconstructing the image to its original size.
- **Sparse Categorical Accuracy**: Unlike standard accuracy, sparse categorical accuracy handles integer class labels without requiring one-hot encoding.
---
## 📘 Chapter 6 – Modern ConvNet, Small Xception Implementation
In this notebook I will explore modern Convnet Architecture, in particular the layers: Residual Connections, Batch Normalization, Separable Convolutions and GlobalAveragePooling2D layer.
Afterwards, by using these 4 layers, I defined and trained a ConvNet Architecture from scratch that looks like a smaller Xception model. This model is used for the Dog Vs. Cat Binary Classification Task to compare the results with the previous models.

### 🔍 What This Covers
- **Residual Connections** Layer.
- **Batch Normalization** Layer.
- **Separable Convolutions** Layer.
- **GlobalAveragePooling2D Layer**.
- Combine these 4 layers to create a **smaller version of Xception  model**.
- This model reached **89% of test accuracy**. While, the simple CNN of Chapter 2 **reached 81%.**
### 🧠 Key Deep Learning Concepts
- **Residual Connections:** These connections add the input of a layer directly to its output, allowing the model to preserve the gradient information.
- **Batch Normalization**: This process normalizes activations within a layer by adjusting and scaling them.
- **Separable Convolutions**: This convolution technique decomposes a traditional convolution into two steps: a depthwise convolution and a pointwise convolution, significantly reducing the computational burden and the number of learnable parameters.
- **GlobalAveragePooling2D Layer**: In modern models, this layer replaces the traditional Flatten Layer. The reason for this is that GlobalAveragePooling2D computes the average of each feature map, reducing the output to a single value per feature, which eliminates the need for a large number of parameters that a Flatten Layer would require.
