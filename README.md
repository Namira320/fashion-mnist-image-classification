# 👕 Fashion MNIST Image Classification Project

This project is a deep learning implementation for image classification using the **Fashion MNIST** dataset. It covers the complete machine learning lifecycle, from data preparation, exploratory data analysis, data augmentation, CNN modeling, training with callbacks, evaluation, to model conversion.

---

## 📚 1. Prerequisites & Libraries
The project requires the following libraries imported in the environment:
* `os`, `json`
* `numpy`
* `matplotlib.pyplot`
* `sklearn.model_selection` (`train_test_split`)
* `tensorflow` & `keras` (`datasets`, `layers`, `models`, `ImageDataGenerator`, `EarlyStopping`, `ReduceLROnPlateau`)

---

## 🗄️ 2. Data Preparation & Preprocessing
* **Data Loading:** The default Fashion-MNIST dataset is loaded via TensorFlow datasets, combined, and structured to a total of 70,000 images.
* **Normalization:** Pixel values are scaled to the range of `[0, 1]` by dividing by `255.0`.
* **Reshaping:** Channel dimensions are expanded to `(28, 28, 1)` for grayscale compatibility.
* **Data Splitting:** Dataset is split into Training, Validation, and Test sets (Train: 44,800, Validation: 11,200, Test: 14,000 samples).
* **Data Augmentation:** Utilizes `ImageDataGenerator` with rotation, width/height shifts, zoom, and horizontal flips to improve generalization.

---

## 🧠 3. Model Architecture
A Convolutional Neural Network (CNN) built using `Sequential` API comprising:
* **Conv2D Layers:** Feature extraction with ReLU activation and padding.
* **Normalization & Pooling:** Batch Normalization, MaxPooling2D, and Dropout layers to prevent overfitting.
* **Global Average Pooling:** Summarizes spatial features.
* **Dense Layers:** Fully connected layers featuring 512 units (ReLU, Dropout) and a final 10-unit output layer with **Softmax** activation for multi-class classification.

---

## 📊 4. Training & Evaluation
* **Callbacks:** `EarlyStopping` and `ReduceLROnPlateau` are implemented to optimize training efficiency.
* **Performance:** Achieved an accuracy of **~88.41%** on the testing dataset.
* **Visualization:** Generates training and validation accuracy/loss plots across epochs.

---

## 🔄 5. Model Conversion
The trained model is exported into multiple deployment formats:
* **Keras / SavedModel format** (`model_fashion.keras`, `saved_model`)
* **TensorFlow Lite (`TFLite`)** (`model.tflite` along with `label.txt`)
* **TensorFlow.js (`tfjs_model`)** with model topology and model weights JSON format.
