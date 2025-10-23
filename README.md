Smile Detection with CNN

This project uses a Convolutional Neural Network (CNN) to detect smiles in real-time using a webcam. The model is trained on the Genki4k dataset, which contains facial images labeled with smile or no-smile categories. Once trained, the model can classify faces captured via webcam and identify whether the person is smiling.

Features

Real-time Smile Detection: Use your webcam to detect smiles in real time.

Training on Custom Dataset: Trains a model using the Genki4k facial image dataset to detect smiles.

Customizable CNN Architecture: The project includes a simple CNN-based model for classification.

Torch and OpenCV: Uses PyTorch for deep learning and OpenCV for real-time webcam capture.

Requirements

Ensure you have the following libraries installed:

torch

torchvision

opencv-python

pandas

scikit-learn

natsort

Pillow

Install dependencies via pip:

pip install torch torchvision opencv-python pandas scikit-learn natsort Pillow 

Setup

Clone the repository:

git clone https://github.com/yourusername/smile-detection.git cd smile-detection 

Prepare the Dataset:

The model is trained using the Genki4k dataset. Download it from here.

Place the dataset's images in the aug_images/ directory.

Create a labels.txt file where each line contains the label for each corresponding image.

Train the Model: The model architecture used is a simple CNN with two convolutional layers, pooling, and two fully connected layers. The training is done using the Adam optimizer with a cross-entropy loss.

python train_model.py 

Test the Model: After training, you can evaluate the model on the test set to check its accuracy.

python test_model.py 

Run the Smile Detector with Webcam: After the model is trained, you can run a real-time smile detection using your webcam:

python smile_detection.py 

Press q to exit the webcam window.

Results

Train Accuracy: 99%

Test Accuracy: 95%

These results were achieved after training the model on the Genki4k dataset for 10 epochs. The model performed exceptionally well on the training data, and the test accuracy remains high, indicating good generalization to unseen data.

Code Overview

train_model.py: This file handles the training of the CNN model using the Genki4k dataset.

test_model.py: Evaluates the model's performance on a test set.

smile_detection.py: Runs the trained model on real-time webcam input, detecting whether a person is smiling.

model.py: Contains the model definition (SmileCNN class).

data_loader.py: Responsible for loading and transforming the dataset for training.

Model Architecture

The SmileCNN model consists of:

Two convolutional layers with ReLU activation and max pooling.

A fully connected layer with 128 units and ReLU activation.

An output layer with 2 units for classifying "smiling" vs "not smiling."

Code Snippet

class SmileCNN(nn.Module): def init(self): super(SmileCNN, self).__init__() self.conv1 = nn.Conv2d(3, 32, 3, padding=1) self.conv2 =
