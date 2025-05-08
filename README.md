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
