## 😊 Smile Detection with CNN

This project uses a Convolutional Neural Network **(CNN)** to detect smiles in real-time using a webcam. The model is trained on the **Genki4k** dataset, which contains facial images labeled with smile or no-smile categories. Once trained, the model can classify faces captured via webcam and identify whether the person is smiling.

---
## 🧩 Part 1 — Preprocessing & Data Augmentation

This part is exactly like **SmileDetection_handcrafted** project. 

All preprocessing scripts are included in `preprocessing.py`.

All augment images scripts are included in `augment_images.py`.

---

## 🌀 Part2 — Train & Test with webcam

Train the Model: The model architecture used is a simple CNN with two convolutional layers, pooling, and two fully connected layers. The training is done using the Adam optimizer with a cross-entropy loss.

---
## 📦 Requirements

Ensure you have the following libraries installed:

* torch

* torchvision

* opencv-python

* pandas

* scikit-learn

* natsort

* Pillow

---

## 🧭 How to Use

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ZahraSafaei1272/SmileDetection_CNN.git
cd SmileDetection_CNN
```
### 2️⃣ Install dependencies

Install dependencies via pip:
```bash  

pip install torch torchvision opencv-python pandas scikit-learn natsort Pillow 
```
### 3️⃣ Run preprocessing
```bash
python preprocessing.py
```
### 4️⃣ Run data augmentation
```bash
python augment_images.py
```
### 5️⃣ Train with CNN & Test the model with webcam

```bash
python smile_detection_with_CNN.py
```
Press q to exit the webcam window.

---
## 📊 Results
### Metric	Value
* Training Accuracy:	0.99
* Testing Accuracy:	0.95

These results were achieved after training the model on the Genki4k dataset for 10 epochs. The model performed exceptionally well on the training data, and the test accuracy remains high, indicating good generalization to unseen data.
