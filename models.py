import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torch.nn.init as init
from visualizer import trainVisualizer
from torcheval.metrics.functional import multiclass_f1_score
from datetime import datetime
import wandb
from sklearn.metrics import confusion_matrix
from visualizer import plotNSaveConfusion
import plotly.io as pio

################
### DATASETS ###
################

class ECGDataset(torch.utils.data.Dataset):
    '''Time series Dataset'''
    def __init__(self, dirPath, experiment, numClasses, classWeights):
        # Root of data
        self.dir = dirPath
        # Its files
        self.fileNames = os.listdir(dirPath)
        # Type of classification task
        self.exp = experiment
        # Number of classes
        self.numClasses = numClasses
        self.weights = classWeights

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        # Create dict to ascribe labels
        nameDict = {f"normal {self.exp}": [1,0,0,0,0], f"AVNRT {self.exp}" : [0,1,0,0,0], 
                    f"AVRT {self.exp}" : [0,0,1,0,0], f"concealed {self.exp}" : [0,0,0,1,0], f"EAT {self.exp}" : [0,0,0,0,1]}
        # Get file
        filePath = os.path.join(self.dir, self.fileNames[idx])
        # Get class label
        fileKey = filePath.split("/")[-1].split("-")[0]
        label = nameDict[fileKey]
        # Load ECG data
        ecgData = np.load(filePath)
        # Reduce data shape (squeeze)
        # Transpose axis to proper format
        # and make it float32, not double
        ecgData = ecgData.transpose(1, 0).astype(np.float32)
        ecgLabels = np.array(label, dtype=np.float32)
        # Make them torch tensors
        ecgData = torch.from_numpy(ecgData)
        ecgLabels = torch.from_numpy(ecgLabels)
        return ecgData, ecgLabels
        
    def samplerWeights(self):
        nameDict = {f"normal {self.exp}": 0, f"AVNRT {self.exp}": 1, f"AVRT {self.exp}": 2, f"concealed {self.exp}": 3, f"EAT {self.exp}": 4}
        # Use the class index from nameDict to get the weight directly from self.weights
        return [self.weights[nameDict[fileName.split("-")[0]]] for fileName in self.fileNames]    


#################################
    
class PTBDataset(torch.utils.data.Dataset):
    '''Time series Dataset'''
    def __init__(self, dirPath, numClasses, classWeights):
        # Root of data
        self.dir = dirPath
        # Its files
        self.fileNames = os.listdir(dirPath)
        # Number of classes
        self.numClasses = numClasses
        # Class weights
        self.weights = classWeights

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        # Create dict to ascribe labels
        nameDict = {f"NORM":[1,0,0,0,0],f"MI":[0,1,0,0,0], 
                    f"STTC":[0,0,1,0,0],f"CD":[0,0,0,1,0],
                    "HYP":[0,0,0,0,1]}
        # Get file
        filePath = os.path.join(self.dir, self.fileNames[idx])
        # Get class label
        fileKey = filePath.split("/")[-1].split("-")[0]
        label = nameDict[fileKey]
        # Load ECG data
        ecgData = np.load(filePath)
        # Expand dims to make it 3D
        ecgData = np.expand_dims(ecgData, axis = -1)
        # Reduce data shape (squeeze)
        # Transpose axis to proper format
        # and make it float32, not double
        ecgData = ecgData.squeeze(-1).transpose(1, 0).astype(np.float32)
        ecgLabels = np.array(label, dtype=np.float32)
        # Make them torch tensors
        ecgData = torch.from_numpy(ecgData)
        ecgLabels = torch.from_numpy(ecgLabels)
        return ecgData, ecgLabels
    

##############
### MODELS ###
##############

def padSeqSymm(batch, targetLength, dimension):
    """Symmetrically pad the sequences in a batch to have a uniform length.
    Params:
    - batch: A batch of input data of shape (batchSize, channels, sequenceLength)
    - targetLength: The desired length for all sequences.
    - dimension: Dimension along which to pad. For our task it's 2."""

    # Calculate the current length of the sequences
    currentLength = batch.size(dimension)
    # Calculate the total padding needed
    totPadNeeded = max(0, targetLength - currentLength)
    # Calculate padding to add to both the start and end of the sequence
    padEachSide = totPadNeeded // 2
    # For an odd padding needed, add the extra padding at the end (+1 at first list, second element)
    if totPadNeeded % 2 == 1:
        padArg = [padEachSide, padEachSide + 1] + [0, 0] * (batch.dim() - dimension - 1)
    # For an even padding needed, add the padding symmetrically
    else:
        padArg = [padEachSide, padEachSide] + [0, 0] * (batch.dim() - dimension - 1)
    # Pad the sequences
    symPadBatch = F.pad(batch, pad=padArg, mode='constant', value=0)
    return symPadBatch


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, targetLength):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=inChannels, out_channels=outChannels, 
                               kernel_size=15, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=outChannels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=outChannels, out_channels=outChannels, 
                               kernel_size=15, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(num_features=outChannels)
        self.relu2 = nn.ReLU()
        self.convRes = nn.Conv1d(in_channels=inChannels, out_channels=outChannels, 
                                 kernel_size=15, stride=2)
        self.bnRes = nn.BatchNorm1d(num_features=outChannels)
        self.relu3 = nn.ReLU()
        self.targetLength = targetLength

    def forward(self, x):
        y = x.clone()
        x = self.bn1(self.conv1(x))
        x = padSeqSymm(x, self.targetLength, 2)
        x = self.relu1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        y = self.bnRes(self.convRes(y))
        y = padSeqSymm(y, self.targetLength, 2)
        x = y + x
        x = self.relu3(x)
        return x


class IDENBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=inChannels, out_channels=outChannels, 
                               kernel_size=15, stride=1, padding='same')
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
    def __init__(self, inChannels, outChannels, targetLength):
        super().__init__()
        self.convBlock = ConvBlock(inChannels, outChannels, targetLength) 
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        if inChannels == outChannels:
            self.IDEN = IDENBlock(inChannels, outChannels)
        else:
            self.IDEN = IDENBlock(inChannels*2, outChannels)
        self.targetLength = targetLength

    def forward(self, x):
        x = self.convBlock(x)
        x = padSeqSymm(x, self.targetLength, 2)
        x = self.pool(x)
        x = padSeqSymm(x, self.targetLength//2, 2)
        x = self.IDEN(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.lin = nn.Linear(inChannels, outChannels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.6)

    def forward(self, x):
        x = self.drop(self.relu(self.lin(x)))
        return x


class ECGCNNClassifier(nn.Module):
    '''Model from paper:
    Automatic multilabel electrocardiogram diagnosis of heart rhythm or conduction abnormalities with deep learning: a cohort study
    It is in the supplementary material'''
    def __init__(self, numClasses):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=15, stride=1, padding='same')
        self.bn = nn.BatchNorm1d(num_features=64)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.convBlock1 = CNNBlock(inChannels=64, outChannels=64, targetLength=625)
        self.convBlock2 = CNNBlock(inChannels=64, outChannels=128, targetLength=156)
        self.convBlock3 = CNNBlock(inChannels=128, outChannels=256, targetLength=39)
        self.convBlock4 = CNNBlock(inChannels=256, outChannels=512, targetLength=9)
        self.avgPool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.Dense1 = DenseBlock(inChannels=1024, outChannels=512)
        self.Dense2 = DenseBlock(inChannels=512, outChannels=512)
        self.fc = nn.Linear(512, numClasses)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.maxPool(self.relu(x))
        x = padSeqSymm(x, 1250, 2)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.avgPool(x)
        x = self.flatten(x)
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
        self.fc = nn.Linear(64, numClasses)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc(x)
        return x


#################
### TRAINING  ###
### FUNCTIONS ###
#################
    

def train(model, trainLoader, valLoader, classes, learningRate, epochs, 
    classWeights, earlyStopPatience, reduceLRPatience, device, expList, 
    dataset, lr, batchSize, L1=None, L2=None):
    '''Training function for the model'''
    print("Starting training")

    model = model.to(device)
    # Get date for saving model weights
    now = datetime.now()
    formattedNow = now.strftime("%d-%m-%y-%H-%M")

    # Add weights to loss and optimizer
    classWeights = torch.from_numpy(classWeights).to(device)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=classWeights)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    # Trackers for callbacks
    bestValF1 = -1
    bestESEpoch = -1
    bestLRepoch = -1

    # Trackers for training visualization
    trainLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List = [], [], [], [], [], []

    ################
    ### TRAINING ###
    ###   LOOP   ###
    ################
    for epoch in range(epochs):
        model.train()
        # Initialize trackers
        trainLoss, correctTrainPreds, totalTrainPreds = 0, 0, 0
        trainPredTensor, trainLabelTensor = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Reset gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            regLoss = 0
            for param in model.parameters():
                if L1 is not None:
                    regLoss += L1 * torch.sum(torch.abs(param))
                if L2 is not None:
                    regLoss += L2 * torch.sum(param ** 2)
            loss += regLoss
            # Backpropagate
            loss.backward()
            optimizer.step()
            # Calculate loss, accuracy and F1 score
            trainLoss += loss.item()
            totalTrainPreds += labels.size(0)
            correctTrainPreds += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()
            trainPredTensor = torch.cat((trainPredTensor, torch.argmax(outputs, 1)))
            trainLabelTensor = torch.cat((trainLabelTensor, torch.argmax(labels, 1)))

        # Validation
        model.eval()
        # Initialize trackers
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


        # Calculate loss and metrics
        epochTrainLoss = trainLoss / len(trainLoader)
        epochTrainAcc = correctTrainPreds / totalTrainPreds * 100
        epochTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=classes, average='macro').item() * 100
        epochValLoss = valLoss / len(valLoader)
        epochValAcc = correctValPreds / totalValPreds * 100
        epochValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=classes, average='macro').item() * 100

        # Print 'em
        print(f'Epoch {epoch+1}: Train Loss: {epochTrainLoss:.4f}, Val Loss: {epochValLoss:.4f}, Train Acc: {epochTrainAcc:.2f}%, Val Accuracy: {epochValAcc:.2f}%, Train F1 Score: {epochTrainF1:.2f}%, Val F1 Score: {epochValF1:.2f}%')
        # Print class F1 scores
        classTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=classes, average=None) * 100
        classValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=classes, average=None) * 100
        for i in range(classTrainF1.size(0)):
            print(f"For class {expList[i].split(" ")[0]} the Train F1: {classTrainF1[i]:.2f}% and Val F1: {classValF1[i]:.2f}%")
            wandb.log({f"Train F1 {expList[i].split(" ")[0]}": classTrainF1[i], f"Val F1 {expList[i].split(" ")[0]}": classValF1[i]}, step=epoch+1, commit=False)

        # Append to visualization trackers
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
            wandb.log({"Best val F1": bestValF1, "Weight Name": f"{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}"}, step=epoch+1, commit=False)
            for i in range(classTrainF1.size(0)):
                wandb.log({f"Best Val F1 {expList[i].split(" ")[0]}": classValF1[i]}, step=epoch+1, commit=False)
            # Save model weights for best validation F1
            torch.save(model.state_dict(), f'/home/tzikos/Desktop/weights/{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}.pth')
            # Create a W&B Table with appropriate column names
            # only for Berts data
            if "AVNRT pre" in expList or "AVNRT tachy" in expList:
                classNames = expList
                # Validation Confusion matrix
                trainPredictions = torch.flatten(trainPredTensor).cpu().numpy()
                trainLabels = torch.flatten(trainLabelTensor).cpu().numpy()
                trainCM = confusion_matrix(trainLabels, trainPredictions)
                # Create a W&B Table with appropriate column names
                # only for Berts data
                columns = ['Class'] + [f'Predicted: {className}' for className in classNames]
                trainCMTable = wandb.Table(columns=columns)
                # Fill the table with data from the confusion matrix
                for i in range(len(trainCM)):
                    row = [classNames[i]] + trainCM[i].tolist()
                    trainCMTable.add_data(*row)
                wandb.log({'Training Confusion Matrix': trainCMTable}, step=epoch+1, commit=False)
                # Plot and save confusion matrix
                figTrain = plotNSaveConfusion(trainCM, classNames, f"/Confusion_train_{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}", "Train")        
                # Convert Plotly figure to an image and log to W&B
                pio.write_image(figTrain, f"/home/tzikos/Confusions/Confusion_train_{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}.png", 
                                width=1000, height=1000)
                wandb.log({"Training Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_train_{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}.png")},
                           step=epoch+1, commit=False)
    
                # Validation Confusion matrix
                valPredictions = torch.flatten(valPredTensor).cpu().numpy()
                valLabels = torch.flatten(valLabelTensor).cpu().numpy()
                valCM = confusion_matrix(valLabels, valPredictions)
                # Create a W&B Table with appropriate column names
                # only for Berts data
                valCMTable = wandb.Table(columns=columns)
                # Fill the table with data from the confusion matrix
                for i in range(len(valCM)):
                    row = [classNames[i]] + valCM[i].tolist()
                    valCMTable.add_data(*row)
                wandb.log({'Validation Confusion Matrix': valCMTable}, step=epoch+1, commit=False)
                # Plot and save confusion matrix
                figVal = plotNSaveConfusion(valCM, classNames, f"/Confusion_val_{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}", "Validation")
                pio.write_image(figVal, f"/home/tzikos/Confusions/Confusion_val_{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}.png", 
                                width=1000, height=1000)
                wandb.log({"Validation Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_val_{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}.png")},
                           step=epoch+1, commit=False)
        if epoch - bestLRepoch > reduceLRPatience - 1:
            learningRate /= 10
            bestLRepoch = epoch
            print("Reducing LR")
        if epoch - bestESEpoch > earlyStopPatience - 1:
            print("Early stopped training at epoch %d" % epoch)
            # terminate the training loop
            break
        wandb.log({"train acc": epochTrainAcc, "train loss": epochTrainLoss, "train F1": epochTrainF1}, step=epoch+1, commit=False)
        wandb.log({"val acc": epochValAcc, "val loss": epochValLoss, "val F1": epochValF1}, step=epoch+1, commit=False)
        print("")
    wandb.log({}, commit=True)
    # Visualize training
    trainVisualizer(trainLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List,
                    saveName=f"Train_hist_{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}")
    # Return the path to the best model weights
    # to be used for testing
    filepath = f'/home/tzikos/Desktop/weights/{model.__class__.__name__}_{dataset}_B{batchSize}_L{lr}_{formattedNow}.pth'
    return filepath

###############
### TESTING ###
###############
def test(model, testLoader, classes, device, filePath, expList):
    print(filePath)
    '''Testing function for the model'''
    # Load model weights
    model.load_state_dict(torch.load(f'{filePath}'))
    model = model.to(device)
    model.eval()
    # Initialize trackers
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
    # Calculate accuracy and F1 score
    testAcc = correctTestPreds / totalTestPreds * 100
    epochTestF1 = multiclass_f1_score(torch.flatten(testPredTensor).long(), torch.flatten(testLabelTensor).long(), num_classes=classes, average='macro').item() * 100
    print(f'Test Accuracy: {testAcc:.2f}%, Test F1 Score: {epochTestF1:.2f}%')
    # Print class F1 scores
    classTestF1 = multiclass_f1_score(torch.flatten(testPredTensor).long(), torch.flatten(testLabelTensor).long(), num_classes=classes, average=None) * 100
    for i in range(classTestF1.size(0)):
        print(f"For class {expList[i].split(" ")[0]} the F1: {classTestF1[i]:.2f}%")
        wandb.log({"Test F1": epochTestF1})
        for i in range(len(expList)):
            wandb.log({f"Test F1 {expList[i].split(" ")[0]}": classTestF1[i]})
    

