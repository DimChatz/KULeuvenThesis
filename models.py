import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import keyboard
import torch.nn.init as init
from visualizer import trainVisualizer
from torcheval.metrics.functional import multiclass_f1_score
from datetime import datetime

################
### DATASETS ###
################

class ECGDataset(torch.utils.data.Dataset):
    '''Time series Dataset'''
    def __init__(self, dirPath, experiment, numClasses, classWeights):
        # Root of data
        self.dir = dirPath
        self.fileNames = os.listdir(dirPath)
        # Type of classification task
        self.exp = experiment
        self.numClasses = numClasses
        self.weights = classWeights

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        # Create dict to ascribe labels
        nameDict = {f"AVNRT {self.exp}" : [1,0,0,0], f"AVRT {self.exp}" : [0,1,0,0], f"concealed {self.exp}" : [0,0,1,0], f"EAT {self.exp}" : [0,0,0,1]}
        # Get file
        filePath = os.path.join(self.dir, self.fileNames[idx])
        # Get file type for label
        fileKey = filePath.split("/")[-1].split("-")[0]
        label = nameDict[fileKey]
        ecgData = np.load(filePath)
        # Reduce data shape (squeeze)
        # Transpose axis to proper format
        # and make it float32, not double
        ecgData = ecgData.squeeze(-1).transpose(1, 0).astype(np.float32)
        ecgLabels = np.array(label, dtype=np.float32)
        ecgData = torch.from_numpy(ecgData)
        ecgLabels = torch.from_numpy(ecgLabels)
        return ecgData, ecgLabels
        
    def samplerWeights(self):
        nameDict = {f"AVNRT {self.exp}": 0, f"AVRT {self.exp}": 1, f"concealed {self.exp}": 2, f"EAT {self.exp}": 3}
        # Use the class index from nameDict to get the weight directly from self.weights
        return [self.weights[nameDict[fileName.split("-")[0]]] for fileName in self.fileNames]

    
    

##############
### MODELS ###
##############

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=inChannels, out_channels=outChannels, kernel_size=15, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=outChannels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=outChannels, out_channels=outChannels, kernel_size=15, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(num_features=outChannels)
        self.relu2 = nn.ReLU()
        self.convRes = nn.Conv1d(in_channels=inChannels, out_channels=outChannels, kernel_size=15, stride=2)
        self.bnRes = nn.BatchNorm1d(num_features=outChannels)
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
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=inChannels, out_channels=outChannels, kernel_size=15, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(num_features=outChannels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = x.clone()
        x = self.relu1(self.bn1(self.conv1(x)))
        x = y + x
        x = self.relu2(x)
        return x

class CNNBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.convBlock = ConvBlock(inChannels, outChannels) 
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        if inChannels == outChannels:
            self.IDEN = IDENBlock(inChannels, outChannels)
        else:
            self.IDEN = IDENBlock(inChannels*2, outChannels)

    def forward(self, x):
        x = self.IDEN(self.pool(self.convBlock(x)))
        return x

class DenseBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.lin = nn.Linear(inChannels, outChannels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.drop(self.relu(self.lin(x)))
        return x

class ECGCNNClassifier(nn.Module):
    '''Model from paper:
    Automatic multilabel electrocardiogram diagnosis of heart rhythm or conduction abnormalities with deep learning: a cohort study
    It is in the supplementary material'''
    def __init__(self, numClasses):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=15, stride=2)
        self.bn = nn.BatchNorm1d(num_features=64)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.convBlock1 = CNNBlock(inChannels=64, outChannels=128)
        self.convBlock2 = CNNBlock(inChannels=128, outChannels=256)
        self.convBlock3 = CNNBlock(inChannels=256, outChannels=512)
        self.avgPool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.Dense1 = DenseBlock(inChannels=1024, outChannels=512)
        self.Dense2 = DenseBlock(inChannels=512, outChannels=512)
        self.fc = nn.Linear(512, numClasses)

    def forward(self, x):
        x = self.maxPool(self.relu(self.bn(self.conv(x))))
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.flatten(self.avgPool(x))
        x = self.Dense1(x)
        x = self.Dense2(x)
        x = self.fc(x)
        return x
    
#############################################

class ECGSimpleClassifier(nn.Module):
    '''Random Model I came up with to see that
        my data runs'''
    def __init__(self, numClasses):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, stride=5)
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=10)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=5)
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.fc1 = nn.Linear(128, 64)
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(64, numClasses)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#################
### TRAINING  ###
### FUNCTIONs ###
#################
def train(model, trainLoader, valLoader, classes, learningRate, epochs, 
    classWeights, earlyStopPatience, reduceLRPatience, device, expList):
    # Model, Loss, and Optimizer
    model = model.to(device)
    now = datetime.now()
    formatted_now = now.strftime("%d-%m-%y-%H-%M")

    # Add weights to loss
    classWeights = torch.from_numpy(classWeights).to(device)
    criterion = nn.CrossEntropyLoss(weight=classWeights)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    # Trackers for callbacks
    bestValF1 = -1
    bestESEpoch = -1
    bestLRepoch = -1

    # Trackers for training vis
    trainLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List = [], [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        trainLoss, correctTrainPreds, totalTrainPreds = 0, 0, 0
        trainPredTensor, trainLabelTensor = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #if torch.equal(labels, torch.from_numpy(np.array([0,0,1,0]))) or torch.equal(labels, torch.from_numpy(np.array([0,0,0,1]))):
            #    print(labels)
            #else:
            #print('outputs are ',torch.argmax(outputs, 1))
            #print('labels are ',torch.argmax(labels, 1))
            #keyboard.wait()
            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            #print(loss)inChannels, outChannels
            optimizer.step()
            # Calculate accuracy
            trainLoss += loss.item()
            totalTrainPreds += labels.size(0)
            #print(labels.size(0))
            correctTrainPreds += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()
            trainPredTensor = torch.cat((trainPredTensor, torch.argmax(outputs, 1)))
            trainLabelTensor = torch.cat((trainLabelTensor, torch.argmax(labels, 1)))
            #print((torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum())
            #keyboard.wait()

        # Validation
        model.eval()
        valLoss, correctValPreds, totalValPreds = 0, 0, 0
        valPredTensor, valLabelTensor = torch.tensor([]).to(device), torch.tensor([]).to(device)
        with torch.no_grad():
            for inputs, labels in valLoader:
                inputs , labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valLoss += loss.item()
                totalValPreds += labels.size(0)
                correctValPreds += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()
                valPredTensor = torch.cat((valPredTensor, torch.argmax(outputs, 1)))
                valLabelTensor = torch.cat((valLabelTensor, torch.argmax(labels, 1)))

        epochTrainLoss = trainLoss / len(trainLoader)
        epochTrainAcc = correctTrainPreds / totalTrainPreds * 100
        epochTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=classes, average='macro').item() * 100
        epochValLoss = valLoss / len(valLoader)
        epochValAcc = correctValPreds / totalValPreds * 100
        epochValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=classes, average='macro').item() * 100

        print(f'Epoch {epoch+1}: Train Loss: {epochTrainLoss:.4f}, Val Loss: {epochValLoss:.4f}, Train Acc: {epochTrainAcc:.2f}%, Val Accuracy: {epochValAcc:.2f}%, Train F1 Score: {epochTrainF1:.2f}%, Val F1 Score: {epochValF1:.2f}%')
        classTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=classes, average=None) * 100
        classValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=classes, average=None) * 100
        for i in range(classTrainF1.size(0)):
            print(f"For class {expList[i].split(" ")[0]} the Train F1: {classTrainF1[i]:.2f}% and Val F1: {classValF1[i]:.2f}%")


        


        # Append to vis trackers
        trainLossList.append(epochTrainLoss)
        valLossList.append(epochValLoss)
        trainAccList.append(epochTrainAcc)
        valAccList.append(epochValAcc)
        trainF1List.append(epochTrainF1)
        valF1List.append(epochValF1)
        
        # Early stopping and reduce LR callbacks
        if epochValF1 > bestValF1:
            bestESEpoch = epoch
            bestLRepoch = epoch
            bestValF1 = epochValF1
            torch.save(model.state_dict(), f'/home/tzikos/Desktop/weights/{model.__class__.__name__}_{formatted_now}.pth')
        if epoch - bestLRepoch > reduceLRPatience - 1:
            learningRate /= 10
            bestLRepoch = epoch
            print("Reducing LR")
        if epoch - bestESEpoch > earlyStopPatience - 1:
            print("Early stopped training at epoch %d" % epoch)
            break  # terminate the training loop

    trainVisualizer(trainLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List)
    filepath = f'/home/tzikos/Desktop/weights/{model.__class__.__name__}_{formatted_now}.pth'
    return filepath

def test(model, testLoader, classes, device, filePath, expList):
    # Test
    model.load_state_dict(torch.load(f'{filePath}'))
    model = model.to(device)
    model.eval()
    correctTestPreds, totalTestPreds = 0, 0
    testPredTensor, testLabelTensor = torch.tensor([]).to(device), torch.tensor([]).to(device)
    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            totalTestPreds += labels.size(0)
            correctTestPreds += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()
            testPredTensor = torch.cat((testPredTensor, torch.argmax(outputs, 1)))
            testLabelTensor = torch.cat((testLabelTensor, torch.argmax(labels, 1)))
    testAcc = correctTestPreds / totalTestPreds * 100
    epochTestF1 = multiclass_f1_score(torch.flatten(testPredTensor).long(), torch.flatten(testLabelTensor).long(), num_classes=classes, average='macro').item() * 100
    print(f'Test Accuracy: {testAcc:.2f}%, Test F1 Score: {epochTestF1:.2f}%')
    classTestF1 = multiclass_f1_score(torch.flatten(testPredTensor).long(), torch.flatten(testLabelTensor).long(), num_classes=classes, average=None) * 100
    for i in range(classTestF1.size(0)):
        print(f"For class {expList[i].split(" ")[0]} the F1: {classTestF1[i]:.2f}%")

