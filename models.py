import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np



class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, dirPath, experiment, numClasses):
        self.dir = dirPath
        self.fileNames = os.listdir(dirPath)
        self.exp = experiment
        self.numClasses = numClasses

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        nameDict = {f"AVNRT {self.exp}" : [1,0,0,0], f"AVRT {self.exp}" : [0,1,0,0], f"concealed {self.exp}" : [0,0,1,0], f"EAT {self.exp}" : [0,0,0,1]}
        filePath = os.path.join(self.dir, self.fileNames[idx])
        fileKey = filePath.split("/")[-1].split("-")[0]
        label = nameDict[fileKey]
        if label is None:
            print(f"Warning: Label for '{filePath}' not found.")
            print(np.load(filePath))
        ecgData = np.load(filePath)
        ecgData = ecgData.squeeze(-1).transpose(1, 0).astype(np.float32)
        ecgLabels = np.array(label, dtype=np.float32)
        ecgData = torch.from_numpy(ecgData)
        ecgLabels = torch.from_numpy(ecgLabels)
        return ecgData, ecgLabels

class ConvBlock(nn.Module):
    def __init__(self, in_channels):
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=15, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=15, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU()
        self.convRes = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=15, stride=2)
        self.bnRes = nn.BatchNorm1d(num_features=64)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        y = x.clone()
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        y = self.bnRes(self.convRes(y))
        x = y + x
        x = self.relu3(x)
        return x


class IDENBlock(nn.Module):
    def __init__(self, in_channels):
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=15, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = x.clone()
        x = self.relu1(self.bn1(self.conv1(x)))
        x = y + x
        return x

class CNNBlock(nn.Module):
    def __init__(self, in_channels):
        self.convBlock = ConvBlock(in_channels=64) 
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.IDEN = IDENBlock(in_channels=in_channels)

    def forward(self, x):
        x = self.IDEN(self.pool1(self.convBlock1(x)))
        return x

class DenseBlock(nn.Module):
    def __init__(self):
        self.lin = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.drop(self.relu(self.lin(x)))
        return x

class ECGCNNClassifier(nn.Module):
    def __init__(self):
        super(ECGCNNClassifier, self).__init__()
        self.conv = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=15, stride=2)
        self.bn = nn.BatchNorm1d(num_features=64)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.convBlock1 = CNNBlock(in_channels=64, out_channels=128)
        self.convBlock2 = CNNBlock(in_channels=64, out_channels=128)
        self.convBlock3 = CNNBlock(in_channels=64, out_channels=128)
        self.convBlock4 = CNNBlock(in_channels=64, out_channels=128)
        self.avgPool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.IDEN1 = IDENBlock()
        self.IDEN2 = IDENBlock()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.maxPool(self.relu(self.bn(self.conv(x))))
        x = self.convBlock1(self.convBlock2(self.convBlock3(self.convBlock4(x))))
        x = self.flatten(self.avgPool(x))
        x = self.IDEN1(self.IDEN2(x))
        x = self.softmax(x)
        return x
    
class ECGSimpleClassifier(nn.Module):
    def __init__(self, numClasses):
        super(ECGSimpleClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, stride=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=10)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, numClasses)  # num_classes is the number of categories

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train(trainLoader, valLoader, testLoader, learningRate, epochs, classes, classWeights):
    # Model, Loss, and Optimizer
    model = ECGSimpleClassifier(classes)
    classWeights = torch.from_numpy(classWeights)
    criterion = nn.CrossEntropyLoss(weight=classWeights)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        trainLoss, correctTrainPreds, totalTrainPreds = 0, 0, 0
        for inputs, labels in trainLoader:
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, torch.max(labels, 1)[1])
            trainLoss += loss.item()
            # Zero the gradients
            optimizer.zero_grad()
            # Perform backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            totalTrainPreds += labels.size(0)
            correctTrainPreds += (predicted == torch.max(labels, 1)[1]).sum().item()

        # Validation
        model.eval()
        valLoss = 0
        correctValPreds = 0
        totalValPreds = 0
        with torch.no_grad():
            for inputs, labels in valLoader:
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(labels, 1)[1])
                valLoss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                totalValPreds += labels.size(0)
                correctValPreds += (predicted == torch.max(labels, 1)[1]).sum().item()

        epochTrainLoss = trainLoss / len(trainLoader)
        epochTrainAcc = correctTrainPreds / totalTrainPreds * 100
        epochValLoss = valLoss / len(valLoader)
        epochValAcc = correctValPreds / totalValPreds * 100
        print(f'Epoch {epoch+1}, Train Loss: {epochTrainLoss:.4f}, Train Accuracy: {epochTrainAcc:.2f}%, Val Loss: {epochValLoss:.4f}, Val Accuracy: {epochValAcc:.2f}%')


    # Test
    model.eval()
    testLoss = 0
    correctTestPreds = 0
    totalTestPreds = 0
    with torch.no_grad():
        for inputs, labels in testLoader:
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            testLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            totalTestPreds += labels.size(0)
            correctTestPreds += (predicted == torch.max(labels, 1)[1]).sum().item()

    testLoss = testLoss / len(testLoader)
    testAcc = correctTestPreds / totalTestPreds * 100
    print(f'Test Loss: {testLoss:.4f}, Test Accuracy: {testAcc:.2f}%')
