import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from visualizer import trainVisualizer
from torcheval.metrics.functional import multiclass_f1_score
from datetime import datetime
import wandb
from sklearn.metrics import confusion_matrix
from visualizer import plotNSaveConfusion
import plotly.io as pio
from torch.utils.data import DataLoader
import time
from torchvision.models import swin_v2_t
from models import lengthFinder, lengthFinderBinary, ECGDataset2_0, ECGCNNClassifier, Gated2TowerTransformer, ECGDataset2_0Binary
from models import MLSTMFCN, CNN2023Attia, PTBDataset
from tqdm import tqdm


#################
### TRAINING  ###
### FUNCTIONS ###
#################
def CVtrain(modelStr, learningRate, epochs, classWeights, earlyStopPatience, 
          reduceLRPatience, expList, dataset, batchSize, L1, L2, usePretrained, 
          modelWeightPath, scaler=1, swin=False, trainNotes=""):
    trainBeginTime = time.time()
    foldF1 = np.zeros((10))
    modelPathList = [None] * 10

    # Get date for saving model weights
    now = datetime.now()
    formattedNow = now.strftime("%d-%m-%y-%H-%M")
    if torch.cuda.is_available():
        print("CUDA is available. Listing devices:")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Running on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model running on {device}")
    experiment = expList[0].split(" ")[-1]
    # Add weights to loss and optimizer
    classWeights = torch.from_numpy(classWeights).to(device)

    for foldNum in range(10):
        if foldNum < 10:
            # start a new wandb run to track this script
            '''Training function for the model'''
            print(f"Started training fold {foldNum+1}")
            trainFileList, valFileList, testFileList = lengthFinder(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/", foldNum)
            if usePretrained:
                # Path to pretrained weights
                pretrainedWeightsPath = f"{modelWeightPath}ECGCNNClassifier_PTBXL_B64_L3e-07_13-03-24-23-28.pth"
                pretrainedWeights = torch.load(pretrainedWeightsPath)
                # Filter out the weights for the classification head
                # Adjusting the key names
                pretrainedWeights = {k: v for k, v in pretrainedWeights.items() if not k.startswith('fc')}
                if modelStr == "CNN2020":
                    model = ECGCNNClassifier(len(classWeights)).to(device)
                    model.load_state_dict(pretrainedWeights, strict=False)
                elif modelStr == "GatedTransformer":
                    model = Gated2TowerTransformer(dimModel=512, dimHidden=2048, dimFeature=12, dimTimestep=5000, 
                                            q=8, v=8, h=8, N=8, classNum=len(classWeights), stage='train', dropout=0.2).to(device)
                    model.load_state_dict(pretrainedWeights, strict=False)
                elif modelStr == "MLSTMFCN":
                    model = MLSTMFCN(len(classWeights)).to(device)
                    model.load_state_dict(pretrainedWeights, strict=False)
            else:
                if modelStr == "CNN2020":
                    model = ECGCNNClassifier(len(classWeights)).to(device)
                elif modelStr == "GatedTransformer":
                    model = Gated2TowerTransformer(dimModel=128, dimHidden=512, dimFeature=12, dimTimestep=5000, 
                                            q=8, v=8, h=8, N=4, classNum=len(classWeights), stage='train', dropout=0.2).to(device)
                elif modelStr == "MLSTMFCN":
                    model = MLSTMFCN(len(classWeights)).to(device)
                elif modelStr == "CNNAttia":
                    model = CNN2023Attia(len(classWeights)).to(device)
                elif modelStr == "swin":
                    model = swin_v2_t()
                    model.head = torch.nn.Linear(model.head.in_features, len(classWeights))
                    model = model.to(device)
            totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total trainable parameters:", totalParams)
            run = wandb.init(
                # set the wandb project where this run will be logged
                project="KU AI Thesis",
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": learningRate,
                "batch_size": batchSize,
                "L1": L1,
                "L2": L2,
                "A": scaler,
                "early_patience": 12,
                "reduce_lr_patience": 5,
                "architecture": model.__class__.__name__,
                "experiment": dataset,
                "use_pretrained": usePretrained}
            )
            wandb.run.notes = trainNotes
            #model.load_state_dict(torch.load("/home/tzikos/Desktop/weights/Models/90-99/ECGCNNClassifier_fold8_tachy_B64_L1e-05_17-04-24-21-01.pth"), strict=False)
            tempRate = learningRate
            criterion = nn.CrossEntropyLoss(weight=classWeights)
            optimizer = optim.Adam(model.parameters(), lr=tempRate)

            # Trackers for callbacks
            bestValF1 = -1
            bestESEpoch = -1
            bestLRepoch = -1
            

            # Trackers for training visualization
            trainLossList, trainVisLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List = [], [], [], [], [], [], []

            trainDataset = ECGDataset2_0(trainFileList, experiment, swin=swin)
            trainLoader = DataLoader(trainDataset, batch_size=batchSize, 
                                shuffle=True, 
                                num_workers=8)
            valDataset = ECGDataset2_0(valFileList, experiment, swin=swin)
            valLoader = DataLoader(valDataset, batch_size=batchSize, 
                                shuffle=True, 
                                num_workers=8)

            ################
            ### TRAINING ###
            ###   LOOP   ###
            ################
            for epoch in range(epochs):
                epochBeginTime = time.time()
                model.train()
                # Initialize trackers
                trainLoss, trainVisLoss, correctTrainPreds, totalTrainPreds = 0, 0, 0, 0
                trainPredTensor, trainLabelTensor = torch.tensor([]).to(device), torch.tensor([]).to(device)
                for inputs, labels in trainLoader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Reset gradients
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    # Calculate loss
                    visLoss = criterion(outputs, labels)
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
                    trainVisLoss += visLoss.item()
                    trainLoss += loss.item()
                    #print(f"Train Loss: {trainLoss:.4f}")
                    #print(f"Train Vis Loss: {trainVisLoss:.4f}")
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
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        valLoss += loss.item()
                        totalValPreds += labels.size(0)
                        correctValPreds += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()
                        valPredTensor = torch.cat((valPredTensor, torch.argmax(outputs, 1)))
                        valLabelTensor = torch.cat((valLabelTensor, torch.argmax(labels, 1)))


                # Calculate loss and metrics
                epochTrainVisLoss = trainVisLoss / len(trainLoader)
                epochTrainLoss = trainLoss / len(trainLoader)
                epochTrainAcc = correctTrainPreds / totalTrainPreds * 100
                epochTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=len(classWeights), average='macro').item() * 100
                epochValLoss = valLoss / len(valLoader)
                epochValAcc = correctValPreds / totalValPreds * 100
                epochValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=len(classWeights), average='macro').item() * 100

                # Print 'em
                print(f'Epoch {epoch+1}: Train Loss: {epochTrainLoss:.4f}, Train Vis Loss: {epochTrainVisLoss:.4f}, Val Loss: {epochValLoss:.4f}, Train Acc: {epochTrainAcc:.2f}%, Val Accuracy: {epochValAcc:.2f}%, Train F1 Score: {epochTrainF1:.2f}%, Val F1 Score: {epochValF1:.2f}%')
                # Print class F1 scores
                classTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=len(classWeights), average=None) * 100
                classValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=len(classWeights), average=None) * 100
                for i in range(classTrainF1.size(0)):
                    print(f"For class {expList[i].split(" ")[0]} the Train F1: {classTrainF1[i]:.2f}% and Val F1: {classValF1[i]:.2f}%")
                    wandb.log({f"Train F1 {expList[i].split(" ")[0]}": classTrainF1[i], f"Val F1 {expList[i].split(" ")[0]}": classValF1[i]}, step=epoch+1, commit=False)

                # Append to visualization trackers
                trainLossList.append(epochTrainLoss)
                trainVisLossList.append(epochTrainVisLoss)
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
                    foldF1[foldNum] = bestValF1
                    modelPathList[foldNum] = f'{modelWeightPath}/{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.pth'
                    wandb.log({"Best val F1": bestValF1, "Weight Name": f"{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}"}, step=epoch+1, commit=False)
                    for i in range(classTrainF1.size(0)):
                        wandb.log({f"Best Val F1 {expList[i].split(" ")[0]}": classValF1[i]}, step=epoch+1, commit=False)
                    # Save model weights for best validation F1
                    torch.save(model.state_dict(), f'{modelWeightPath}/{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.pth')
                    # Create a W&B Table with appropriate column names
                    # only for Berts data
                    if f"AVNRT {experiment}" in expList:
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
                        figTrain = plotNSaveConfusion(trainCM, classNames, f"/Confusion_train_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}", "Train")        
                        # Convert Plotly figure to an image and log to W&B
                        pio.write_image(figTrain, f"/home/tzikos/Confusions/Confusion_train_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png", 
                                        width=1000, height=1000)
                        wandb.log({"Training Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_train_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png")},
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
                        figVal = plotNSaveConfusion(valCM, classNames, f"/Confusion_val_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}", "Validation")
                        pio.write_image(figVal, f"/home/tzikos/Confusions/Confusion_val_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png", 
                                        width=1000, height=1000)
                        wandb.log({"Validation Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_val_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png")},
                                step=epoch+1, commit=False)
                if epoch - bestLRepoch > reduceLRPatience - 1:
                    tempRate /= 10
                    bestLRepoch = epoch
                    print("Reducing LR")
                if epoch - bestESEpoch > earlyStopPatience - 1:
                    print("Early stopped training at epoch %d" % epoch)
                    # terminate the training loop
                    break
                wandb.log({"train acc": epochTrainAcc, "train loss": epochTrainLoss, "train vis loss": epochTrainVisLoss, "train F1": epochTrainF1}, step=epoch+1, commit=False)
                wandb.log({"val acc": epochValAcc, "val loss": epochValLoss, "val F1": epochValF1}, step=epoch+1, commit=False)
                epochEndTime = time.time()
                epochTime = (epochEndTime - epochBeginTime) / 60
                print(f"Time taken for epoch {epoch+1}: {epochTime:.2f} minutes")
                print("")
            # Visualize training
            trainVisualizer(trainVisLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List,
                            saveName=f"Train_hist_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}")
        ###############
            ### TESTING ###
            ############### 
            '''Testing function for the model'''
            # Load model weights

            if modelStr == "CNN2020":
                model = ECGCNNClassifier(len(classWeights))
            elif modelStr == "GatedTransformer":
                model = Gated2TowerTransformer(dimModel=128, dimHidden=512, dimFeature=12, dimTimestep=5000, 
                        q=8, v=8, h=8, N=4, classNum=len(classWeights), stage='test', dropout=0.2)
            elif modelStr == "MLSTMFCN":
                model = MLSTMFCN(len(classWeights))
            elif modelStr == "CNNAttia":
                model = CNN2023Attia(len(classWeights))
            elif modelStr == "swin":
                model = swin_v2_t()
                model.head = torch.nn.Linear(model.head.in_features, len(classWeights))        
            model.load_state_dict(torch.load(f'{modelPathList[foldNum]}'))
            model = model.to(device)
            model.eval()
            # Initialize trackers
            correctTestPreds, totalTestPreds = 0, 0
            testPredTensor, testLabelTensor = torch.tensor([]).to(device), torch.tensor([]).to(device)


            testDataset = ECGDataset2_0(testFileList, expList[0].split(" ")[-1], swin=swin)
            testLoader = DataLoader(testDataset, batch_size=128, 
                                    shuffle=True, 
                                    num_workers=8)
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
            epochTestF1 = multiclass_f1_score(torch.flatten(testPredTensor).long(), torch.flatten(testLabelTensor).long(), num_classes=len(expList), average='macro').item() * 100
            print(f'Test Accuracy: {testAcc:.2f}%, Test F1 Score: {epochTestF1:.2f}%')
            # Print class F1 scores
            classTestF1 = multiclass_f1_score(torch.flatten(testPredTensor).long(), torch.flatten(testLabelTensor).long(), num_classes=len(expList), average=None) * 100
            for i in range(classTestF1.size(0)):
                print(f"For class {expList[i].split(" ")[0]} the F1: {classTestF1[i]:.2f}%")
                wandb.log({"Test F1": epochTestF1}, commit=False)
                for i in range(len(expList)):
                    wandb.log({f"Test F1 {expList[i].split(" ")[0]}": classTestF1[i]}, commit=False)
            # Create a W&B Table with appropriate column names
            # only for Berts data
            if f"AVNRT {experiment}" in expList:
                classNames = expList
                # Validation Confusion matrix
                testPredictions = torch.flatten(testPredTensor).cpu().numpy()
                testLabels = torch.flatten(testLabelTensor).cpu().numpy()
                testCM = confusion_matrix(testLabels, testPredictions)
                # Create a W&B Table with appropriate column names
                # only for Berts data
                columns = ['Class'] + [f'Predicted: {className}' for className in classNames]
                testCMTable = wandb.Table(columns=columns)
                # Fill the table with data from the confusion matrix
                for i in range(len(testCM)):
                    row = [classNames[i]] + testCM[i].tolist()
                    testCMTable.add_data(*row)
                wandb.log({'Testing Confusion Matrix': testCMTable}, commit=False)
                # Plot and save confusion matrix
                figTest = plotNSaveConfusion(testCM, classNames, f"/Confusion_test_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}", "Test")        
                # Convert Plotly figure to an image and log to W&B
                pio.write_image(figTest, f"/home/tzikos/Confusions/Confusion_test_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png", 
                                width=1000, height=1000)
                wandb.log({"Testing Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_test_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png")},
                        commit=False)
            
            wandb.log({}, commit=True)
            run.finish()
            trainEndTime = time.time()
            runTime = (trainEndTime - trainBeginTime) / 60
            print(f"Time taken for training : {runTime:.2f} minutes")

#######################################################################3

def train(modelStr, learningRate, classWeights, expList, batchSize, modelWeightPath, L1, L2,
          epochs=1000, earlyStopPatience=12, reduceLRPatience=5, swin=False, trainNotes="Pretraining PTB"):
    trainBeginTime = time.time()
    # Get date for saving model weights
    now = datetime.now()
    formattedNow = now.strftime("%d-%m-%y-%H-%M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model running on {device}")
    dataset = "ptb"
    # Add weights to loss and optimizer
    classWeights = torch.from_numpy(classWeights).to(device)
    # start a new wandb run to track this script
    '''Training function for the model'''
    print(f"Starting training")
    if modelStr == "CNN2020":
        model = ECGCNNClassifier(len(classWeights)).to(device)
    elif modelStr == "GatedTransformer":
        model = Gated2TowerTransformer(dimModel=128, dimHidden=512, dimFeature=12, dimTimestep=5000, 
                                    q=8, v=8, h=8, N=4, classNum=len(classWeights), stage='train', dropout=0.2).to(device)
    elif modelStr == "MLSTMFCN":
        model = MLSTMFCN(len(classWeights)).to(device)
    elif modelStr == "CNNAttia":
        model = CNN2023Attia(len(classWeights)).to(device)
    elif modelStr == "swin":
        model = swin_v2_t()
        model.head = torch.nn.Linear(model.head.in_features, len(classWeights))
        model = model.to(device)
    totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", totalParams)
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="KU AI Thesis",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": learningRate,
        "batch_size": batchSize,
        "early_patience": 12,
        "reduce_lr_patience": 5,
        "architecture": model.__class__.__name__,
        "experiment": dataset}
    )
    wandb.run.notes = trainNotes
    model.load_state_dict(torch.load("/home/tzikos/Desktop/weights/Models/90-99/ECGCNNClassifier_fold8_tachy_B64_L1e-05_17-04-24-21-01.pth"), strict=False)
    criterion = nn.CrossEntropyLoss(weight=classWeights)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    # Trackers for callbacks
    bestValF1 = -1
    bestESEpoch = -1
    bestLRepoch = -1
    # Trackers for training visualization
    trainLossList, trainVisLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List = [], [], [], [], [], [], []

    trainDataset = PTBDataset(f"/home/tzikos/Desktop/Data/PTBXL Diagnostic torch/train/", 5, swin=swin)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=8)
    valDataset = PTBDataset(f"/home/tzikos/Desktop/Data/PTBXL Diagnostic torch/val/", 5, swin=swin)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True, num_workers=8)

    ################
    ### TRAINING ###
    ###   LOOP   ###
    ################
    for epoch in range(epochs):
        epochBeginTime = time.time()
        model.train()
        # Initialize trackers
        trainLoss, trainVisLoss, correctTrainPreds, totalTrainPreds = 0, 0, 0, 0
        trainPredTensor, trainLabelTensor = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Reset gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            # Calculate loss
            visLoss = criterion(outputs, labels)
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
            trainVisLoss += visLoss.item()
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
        epochTrainVisLoss = trainVisLoss / len(trainLoader)
        epochTrainLoss = trainLoss / len(trainLoader)
        epochTrainAcc = correctTrainPreds / totalTrainPreds * 100
        epochTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=len(classWeights), average='macro').item() * 100
        epochValLoss = valLoss / len(valLoader)
        epochValAcc = correctValPreds / totalValPreds * 100
        epochValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=len(classWeights), average='macro').item() * 100

        # Print 'em
        print(f'Epoch {epoch+1}: Train Loss: {epochTrainLoss:.4f}, Train Vis Loss: {epochTrainVisLoss:.4f}, Val Loss: {epochValLoss:.4f}, Train Acc: {epochTrainAcc:.2f}%, Val Accuracy: {epochValAcc:.2f}%, Train F1 Score: {epochTrainF1:.2f}%, Val F1 Score: {epochValF1:.2f}%')
        # Print class F1 scores
        classTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=len(classWeights), average=None) * 100
        classValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=len(classWeights), average=None) * 100
        for i in range(classTrainF1.size(0)):
            print(f"For class {expList[i].split(" ")[0]} the Train F1: {classTrainF1[i]:.2f}% and Val F1: {classValF1[i]:.2f}%")
            wandb.log({f"Train F1 {expList[i].split(" ")[0]}": classTrainF1[i], f"Val F1 {expList[i].split(" ")[0]}": classValF1[i]}, step=epoch+1, commit=False)

        # Append to visualization trackers
        trainLossList.append(epochTrainLoss)
        trainVisLossList.append(epochTrainVisLoss)
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
            wandb.log({"Best val F1": bestValF1, "Weight Name": f"{model.__class__.__name__}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}"}, step=epoch+1, commit=False)
            for i in range(classTrainF1.size(0)):
                wandb.log({f"Best Val F1 {expList[i].split(" ")[0]}": classValF1[i]}, step=epoch+1, commit=False)
            # Save model weights for best validation F1
            torch.save(model.state_dict(), f'{modelWeightPath}/{model.__class__.__name__}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.pth')
        if epoch - bestLRepoch > reduceLRPatience - 1:
            learningRate /= 10
            bestLRepoch = epoch
            print("Reducing LR")
        if epoch - bestESEpoch > earlyStopPatience - 1:
            print("Early stopped training at epoch %d" % epoch)
            # terminate the training loop
            break
        wandb.log({"train acc": epochTrainAcc, "train loss": epochTrainLoss, "train vis loss": epochTrainVisLoss, "train F1": epochTrainF1}, step=epoch+1, commit=False)
        wandb.log({"val acc": epochValAcc, "val loss": epochValLoss, "val F1": epochValF1}, step=epoch+1, commit=False)
        epochEndTime = time.time()
        epochTime = (epochEndTime - epochBeginTime) / 60
        print(f"Time taken for epoch {epoch+1}: {epochTime:.2f} minutes")
        print("")
    # Visualize training
    trainVisualizer(trainVisLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List,
                    saveName=f"Train_hist_{model.__class__.__name__}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}") 
    wandb.log({}, commit=True)
    run.finish()
    trainEndTime = time.time()
    runTime = (trainEndTime - trainBeginTime) / 60
    print(f"Time taken for training : {runTime:.2f} minutes")



def CVtrainBinary(modelStr, learningRate, epochs, classWeights, earlyStopPatience,
          reduceLRPatience, expList, dataset, batchSize, L1, L2, usePretrained,
          modelWeightPath, scaler=1, swin=False, trainNotes="", AVRT=False):
    trainBeginTime = time.time()
    foldF1 = np.zeros((10))
    modelPathList = [None] * 10

    # Get date for saving model weights
    now = datetime.now()
    formattedNow = now.strftime("%d-%m-%y-%H-%M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model running on {device}")
    experiment = expList[0].split(" ")[-1]
    # Add weights to loss and optimizer
    classWeights = torch.from_numpy(classWeights).to(device)

    for foldNum in range(10):
        if foldNum < 10:
                # start a new wandb run to track this script
            '''Training function for the model'''
            print(f"Starting training fold {foldNum+1}")
            if AVRT:
                trainFileList, valFileList, testFileList = lengthFinderBinary(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/", foldNum)
            else:
                trainFileList, valFileList, testFileList = lengthFinder(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/", foldNum)
            if usePretrained:
                # Path to pretrained weights
                pretrainedWeightsPath = f"{modelWeightPath}ECGCNNClassifier_PTBXL_B64_L3e-07_13-03-24-23-28.pth"
                pretrainedWeights = torch.load(pretrainedWeightsPath)
                # Filter out the weights for the classification head
                # Adjusting the key names
                pretrainedWeights = {k: v for k, v in pretrainedWeights.items() if not k.startswith('fc')}
                if modelStr == "CNN2020":
                    model = ECGCNNClassifier(len(classWeights)).to(device)
                    model.load_state_dict(pretrainedWeights, strict=False)
                elif modelStr == "GatedTransformer":
                    model = Gated2TowerTransformer(dimModel=512, dimHidden=2048, dimFeature=12, dimTimestep=5000, 
                                            q=8, v=8, h=8, N=8, classNum=len(classWeights), stage='train', dropout=0.2).to(device)
                    model.load_state_dict(pretrainedWeights, strict=False)
                elif modelStr == "MLSTMFCN":
                    model = MLSTMFCN(len(classWeights)).to(device)
                    model.load_state_dict(pretrainedWeights, strict=False)
            else:
                if modelStr == "CNN2020":
                    model = ECGCNNClassifier(len(classWeights)).to(device)
                elif modelStr == "GatedTransformer":
                    model = Gated2TowerTransformer(dimModel=128, dimHidden=512, dimFeature=12, dimTimestep=5000, 
                                            q=8, v=8, h=8, N=4, classNum=len(classWeights), stage='train', dropout=0.2).to(device)
                elif modelStr == "MLSTMFCN":
                    model = MLSTMFCN(len(classWeights)).to(device)
                elif modelStr == "CNNAttia":
                    model = CNN2023Attia(len(classWeights)).to(device)
                elif modelStr == "swin":
                    model = swin_v2_t()
                    model.head = torch.nn.Linear(model.head.in_features, len(classWeights))
                    model = model.to(device)
            totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total trainable parameters:", totalParams)
            run = wandb.init(
                # set the wandb project where this run will be logged
                project="KU AI Thesis",
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": learningRate,
                "batch_size": batchSize,
                "L1": L1,
                "L2": L2,
                "A": scaler,
                "early_patience": earlyStopPatience,
                "reduce_lr_patience": reduceLRPatience,
                "architecture": model.__class__.__name__,
                "experiment": dataset,
                "use_pretrained": usePretrained}
            )
            wandb.run.notes = trainNotes
            #model.load_state_dict(torch.load("/home/tzikos/Desktop/weights/Models/90-99/ECGCNNClassifier_fold8_tachy_B64_L1e-05_17-04-24-21-01.pth"), strict=False)
            tempRate = learningRate
            criterion = nn.CrossEntropyLoss(weight=classWeights)
            optimizer = optim.Adam(model.parameters(), lr=tempRate)

            # Trackers for callbacks
            bestValF1 = -1
            bestESEpoch = -1
            bestLRepoch = -1
            

            # Trackers for training visualization
            trainLossList, trainVisLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List = [], [], [], [], [], [], []
            trainDataset = ECGDataset2_0Binary(trainFileList, experiment, swin=swin, AVRT=AVRT)
            trainLoader = DataLoader(trainDataset, batch_size=batchSize, 
                                shuffle=True, 
                                num_workers=8)
            valDataset = ECGDataset2_0Binary(valFileList, experiment, swin=swin, AVRT=AVRT)
            valLoader = DataLoader(valDataset, batch_size=batchSize, 
                                shuffle=True, 
                                num_workers=8)

            ################
            ### TRAINING ###
            ###   LOOP   ###
            ################
            for epoch in range(epochs):
                epochBeginTime = time.time()
                model.train()
                # Initialize trackers
                trainLoss, trainVisLoss, correctTrainPreds, totalTrainPreds = 0, 0, 0, 0
                trainPredTensor, trainLabelTensor = torch.tensor([]).to(device), torch.tensor([]).to(device)
                for inputs, labels in trainLoader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Reset gradients
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    # Calculate loss
                    visLoss = criterion(outputs, labels)
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
                    trainVisLoss += visLoss.item()
                    trainLoss += loss.item()
                    #print(f"Train Loss: {trainLoss:.4f}")
                    #print(f"Train Vis Loss: {trainVisLoss:.4f}")
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
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        valLoss += loss.item()
                        totalValPreds += labels.size(0)
                        correctValPreds += (torch.argmax(outputs, 1) == torch.argmax(labels, 1)).sum().item()
                        valPredTensor = torch.cat((valPredTensor, torch.argmax(outputs, 1)))
                        valLabelTensor = torch.cat((valLabelTensor, torch.argmax(labels, 1)))


                # Calculate loss and metrics
                epochTrainVisLoss = trainVisLoss / len(trainLoader)
                epochTrainLoss = trainLoss / len(trainLoader)
                epochTrainAcc = correctTrainPreds / totalTrainPreds * 100
                epochTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=len(classWeights), average='macro').item() * 100
                epochValLoss = valLoss / len(valLoader)
                epochValAcc = correctValPreds / totalValPreds * 100
                epochValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=len(classWeights), average='macro').item() * 100

                # Print 'em
                print(f'Epoch {epoch+1}: Train Loss: {epochTrainLoss:.4f}, Train Vis Loss: {epochTrainVisLoss:.4f}, Val Loss: {epochValLoss:.4f}, Train Acc: {epochTrainAcc:.2f}%, Val Accuracy: {epochValAcc:.2f}%, Train F1 Score: {epochTrainF1:.2f}%, Val F1 Score: {epochValF1:.2f}%')
                # Print class F1 scores
                classTrainF1 = multiclass_f1_score(torch.flatten(trainPredTensor).long(), torch.flatten(trainLabelTensor).long(), num_classes=len(classWeights), average=None) * 100
                classValF1 = multiclass_f1_score(torch.flatten(valPredTensor).long(), torch.flatten(valLabelTensor).long(), num_classes=len(classWeights), average=None) * 100
                for i in range(classTrainF1.size(0)):
                    print(f"For class {expList[i].split(" ")[0]} the Train F1: {classTrainF1[i]:.2f}% and Val F1: {classValF1[i]:.2f}%")
                    wandb.log({f"Train F1 {expList[i].split(" ")[0]}": classTrainF1[i], f"Val F1 {expList[i].split(" ")[0]}": classValF1[i]}, step=epoch+1, commit=False)

                # Append to visualization trackers
                trainLossList.append(epochTrainLoss)
                trainVisLossList.append(epochTrainVisLoss)
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
                    foldF1[foldNum] = bestValF1
                    modelPathList[foldNum] = f'{modelWeightPath}/{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.pth'
                    wandb.log({"Best val F1": bestValF1, "Weight Name": f"{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}"}, step=epoch+1, commit=False)
                    for i in range(classTrainF1.size(0)):
                        wandb.log({f"Best Val F1 {expList[i].split(" ")[0]}": classValF1[i]}, step=epoch+1, commit=False)
                    # Save model weights for best validation F1
                    torch.save(model.state_dict(), f'{modelWeightPath}/{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.pth')
                    # Create a W&B Table with appropriate column names
                    # only for Berts data
                    if f"normal {experiment}" in expList or f"AVNRT {experiment}" in expList:
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
                        figTrain = plotNSaveConfusion(trainCM, classNames, f"/Confusion_train_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}", "Train")        
                        # Convert Plotly figure to an image and log to W&B
                        pio.write_image(figTrain, f"/home/tzikos/Confusions/Confusion_train_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png", 
                                        width=1000, height=1000)
                        wandb.log({"Training Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_train_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png")},
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
                        figVal = plotNSaveConfusion(valCM, classNames, f"/Confusion_val_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}", "Validation")
                        pio.write_image(figVal, f"/home/tzikos/Confusions/Confusion_val_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png", 
                                        width=1000, height=1000)
                        wandb.log({"Validation Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_val_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png")},
                                step=epoch+1, commit=False)
                if epoch - bestLRepoch > reduceLRPatience - 1:
                    tempRate /= 10
                    bestLRepoch = epoch
                    print("Reducing LR")
                if epoch - bestESEpoch > earlyStopPatience - 1:
                    print("Early stopped training at epoch %d" % epoch)
                    # terminate the training loop
                    break
                wandb.log({"train acc": epochTrainAcc, "train loss": epochTrainLoss, "train vis loss": epochTrainVisLoss, "train F1": epochTrainF1}, step=epoch+1, commit=False)
                wandb.log({"val acc": epochValAcc, "val loss": epochValLoss, "val F1": epochValF1}, step=epoch+1, commit=False)
                epochEndTime = time.time()
                epochTime = (epochEndTime - epochBeginTime) / 60
                print(f"Time taken for epoch {epoch+1}: {epochTime:.2f} minutes")
                print("")
            # Visualize training
            trainVisualizer(trainVisLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List,
                            saveName=f"Train_hist_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}")
            ###############
            ### TESTING ###
            ############### 
            '''Testing function for the model'''
            # Load model weights

            if modelStr == "CNN2020":
                model = ECGCNNClassifier(len(classWeights))
            elif modelStr == "GatedTransformer":
                model = Gated2TowerTransformer(dimModel=128, dimHidden=512, dimFeature=12, dimTimestep=5000, 
                        q=8, v=8, h=8, N=4, classNum=len(classWeights), stage='test', dropout=0.2)
            elif modelStr == "MLSTMFCN":
                model = MLSTMFCN(len(classWeights))
            elif modelStr == "CNNAttia":
                model = CNN2023Attia(len(classWeights))
            elif modelStr == "swin":
                model = swin_v2_t()
                model.head = torch.nn.Linear(model.head.in_features, len(classWeights))        
            model.load_state_dict(torch.load(f'{modelPathList[foldNum]}'))
            model = model.to(device)
            model.eval()
            # Initialize trackers
            correctTestPreds, totalTestPreds = 0, 0
            testPredTensor, testLabelTensor = torch.tensor([]).to(device), torch.tensor([]).to(device)


            testDataset = ECGDataset2_0Binary(testFileList, experiment, swin=swin, AVRT=AVRT)
            testLoader = DataLoader(testDataset, batch_size=batchSize, 
                                shuffle=True, 
                                num_workers=8)
            
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
            epochTestF1 = multiclass_f1_score(torch.flatten(testPredTensor).long(), torch.flatten(testLabelTensor).long(), num_classes=len(expList), average='macro').item() * 100
            print(f'Test Accuracy: {testAcc:.2f}%, Test F1 Score: {epochTestF1:.2f}%')
            # Print class F1 scores
            classTestF1 = multiclass_f1_score(torch.flatten(testPredTensor).long(), torch.flatten(testLabelTensor).long(), num_classes=len(expList), average=None) * 100
            for i in range(classTestF1.size(0)):
                print(f"For class {expList[i].split(" ")[0]} the F1: {classTestF1[i]:.2f}%")
                wandb.log({"Test F1": epochTestF1}, commit=False)
                for i in range(len(expList)):
                    wandb.log({f"Test F1 {expList[i].split(" ")[0]}": classTestF1[i]}, commit=False)
            # Create a W&B Table with appropriate column names
            # only for Berts data
            if f"normal {experiment}" in expList or f"AVNRT {experiment}" in expList:
                classNames = expList
                # Validation Confusion matrix
                testPredictions = torch.flatten(testPredTensor).cpu().numpy()
                testLabels = torch.flatten(testLabelTensor).cpu().numpy()
                testCM = confusion_matrix(testLabels, testPredictions)
                # Create a W&B Table with appropriate column names
                # only for Berts data
                columns = ['Class'] + [f'Predicted: {className}' for className in classNames]
                testCMTable = wandb.Table(columns=columns)
                # Fill the table with data from the confusion matrix
                for i in range(len(testCM)):
                    row = [classNames[i]] + testCM[i].tolist()
                    testCMTable.add_data(*row)
                wandb.log({'Testing Confusion Matrix': testCMTable}, commit=False)
                # Plot and save confusion matrix
                figTest = plotNSaveConfusion(testCM, classNames, f"/Confusion_test_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}", "Test")        
                # Convert Plotly figure to an image and log to W&B
                pio.write_image(figTest, f"/home/tzikos/Confusions/Confusion_test_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png", 
                                width=1000, height=1000)
                wandb.log({"Testing Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_test_{model.__class__.__name__}_fold{foldNum+1}_{dataset}_B{batchSize}_L{learningRate}_{formattedNow}.png")},
                        commit=False)
            
            wandb.log({}, commit=True)
            run.finish()
            trainEndTime = time.time()
            runTime = (trainEndTime - trainBeginTime) / 60
            print(f"Time taken for training : {runTime:.2f} minutes")