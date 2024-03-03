import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np

# Hyperparameters
LEARNING_RATE = 0.001
BATCH = 64
EPOCHS = 25
CLASSES = 4
EXPERIMENT = "pre"
DIR_PATH = f"/home/tzikos/Desktop/Data/Berts torch/{EXPERIMENT}/"


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, dirPath=DIR_PATH):
        self.dir = dirPath
        self.fileNames = os.listdir(dirPath)

    def __len__(self):
        return len(self.dir)

    def __getitem__(self, idx):
        nameDict = {"AVNRT" : 1, "AVRT" : 2, "concealed" : 3, "EAT" : 4}
        filePath = os.path.join(self.dir, self.fileNames[idx])
        ecgData = np.load(filePath)
        label = nameDict[self.fileNames[idx].split(".")[0]]
        ecgLabels = np.full(shape = (ecgData.shape[2],), fill_value = label)
        ecgData = torch.from_numpy(ecgData)
        ecgLabels = torch.from_numpy(ecgLabels)
        return 
    
class ECGClassifier(nn.Module):
    def __init__(self):
        super(ECGClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, CLASSES)  # num_classes is the number of categories

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


# Dataset
train_dataset = ECGDataset(DIR_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=-1)

# Model, Loss, and Optimizer
model = ECGClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform backward pass
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions * 100
    
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
