import glob
import cv2
import os
from natsort import natsorted
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Paths
root_dir = "SmileDetection_CNN"
label_file = os.path.join(root_dir, "labels.txt")

image_paths = natsorted(glob.glob("aug_images/*.jpg"))



labels = []
with open(label_file, 'r') as f:
    for line in f:
        element = line.split()[0]
        labels.append(int(element))
labels = np.repeat(labels, 5)  
# Convert numpy array → torch tensor with dtype long
labels = torch.tensor(labels, dtype=torch.long)


train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)
print(len(train_paths), "train samples,", len(test_paths), "test samples")



transform = transforms.Compose([
    transforms.Resize((64,64)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])



class Genki4kDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label



train_dataset = Genki4kDataset(train_paths, train_labels, transform=transform)
test_dataset = Genki4kDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



class SmileCNN(nn.Module):
    def __init__(self):
        super(SmileCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*16*16, 128)  # 64x16x16 after pooling
        self.fc2 = nn.Linear(128, 2)  # 2 classes: smile / no smile

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32,32,32)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64,16,16)
        x = x.view(-1, 64*16*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmileCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(10):
    model.train()
    running_loss, correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = correct / len(train_dataset)
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Train Acc: {acc:.4f}")



model.eval()
correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        correct += (outputs.argmax(1) == labels).sum().item()

test_acc = correct / len(test_dataset)
print("Test Accuracy:", test_acc)




#Test with Webcam




import cv2




# Define same transform used in training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # match training
])

model.eval()
cap=cv2.VideoCapture(0)
while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if ret:
    # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using a face detector
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_classifier.detectMultiScale(gray, 1.1, 4)

        # Loop over all detected faces
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # crop face
            face_tensor = transform(face)  # (3, 64, 64)
            face_tensor = face_tensor.unsqueeze(0)  # (1, 3, 64, 64)
            
            with torch.no_grad():
                prediction = model(face_tensor)
                predicted_class = torch.argmax(prediction, dim=1).item()

            if predicted_class == 1:  # smiling
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Smiling', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Smile Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()





