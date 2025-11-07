# Real Time Indian Sign Language Detection

## Overview

This project captures live video from a webcam, extracts **hand landmarks**, and classifies them into **ISL gestures** using a trained neural network. It is designed to be:

* Real-time
* Lightweight
* Student-friendly
* Perfect for final-year projects and demos

---

##  Features

* Live ISL detection using webcam
* Supports multiple gestures (A–Z / digits / custom signs)
* Uses **MediaPipe Hands** for landmark extraction
* Deep Learning model for classification
* Smooth predictions with buffer averaging
* Modular and easy to extend

---

##  Tech Stack

* **Python 3.x**
* **OpenCV** – video capture
* **MediaPipe** – hand tracking
* **TensorFlow / PyTorch** – model
* **NumPy** – data processing
* **Pickle** – label encoding

---

##  System Architecture

1. **Webcam Frame Capture**
2. **Hand Landmark Detection (MediaPipe)**
3. **Landmark Preprocessing (Normalization + Flattening)**
4. **Deep Learning Model Prediction**
5. **Prediction Smoothing (Deque Buffer)**
6. **Display Output on Screen**

---
<img width="723" height="381" alt="image" src="https://github.com/user-attachments/assets/fc1dc48f-3ac9-436a-b3f5-b6188afe425a" />

##  Model Training Summary

* Landmark data collected using MediaPipe
* Preprocessed into (42/63) keypoints per frame
* Trained CNN/RNN/MLP model
* Saved weights + encoder

Example metrics:

* **Accuracy: ~93%**
* **Supports multiple classes**

---
