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
