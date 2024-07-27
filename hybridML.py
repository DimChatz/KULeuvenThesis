""" 
This is a hybrid machine learning model pipeline that takes as input the embeddings
of an pretrained model and uses them as input to an LDA model.
The output then is followed by an input to a SVM model.
"""
import torch
import time
from models import foldFinder
import re
import wandb
from models import ECGDataset2_0, ECGDataset2_0Binary
from torch.utils.data import DataLoader
from utility import ModelWithoutLastLayer, seedEverything
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import plotly.io as pio
from visualizer import plotNSaveConfusion
from datetime import datetime
from models import CNN2023Attia, ECGCNNClassifier, MLSTMFCN, Gated2TowerTransformer
import os


def createNsaveEmbeddings(model, experiment:str, modelName:str,
                          batchSize:int, device:torch.device, loader:DataLoader,
                          subtype:str, fold:int, channels:int=12, timeSteps:int=5000):
    embeddingSize = model(torch.zeros((batchSize, channels, timeSteps)).to(device))
    #print(f"Embedding size is {embeddingSize.size()}")
    allEmbeddings = torch.empty(0, embeddingSize.size(1)).to(device)
    allLabels = torch.empty(0).to(device)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            #print(f"Inputs size is {inputs.size()}")
            outputs = model(inputs)
            #print(f"Outputs size is {outputs.size()}")
            allEmbeddings = torch.cat((allEmbeddings, outputs))
            allLabels = torch.cat((allLabels, torch.argmax(labels,1)))
    allEmbeddings = allEmbeddings.cpu().numpy()
    #print(f"Embeddings shape is {allEmbeddings.shape}")
    os.makedirs(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/", exist_ok=True)
    np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{modelName}embeddings.npy", allEmbeddings)
    allLabels = allLabels.cpu().numpy()
    #print(f"Labels shape is {allLabels.shape}")
    np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{modelName}labels.npy", allLabels)


def preprocEmbed(model: torch.nn.Module, modelName: str,  weightsPath: str, experiment: str,
            batchSize: int, subtype:str, fold:int):
    path=f"/home/tzikos/Desktop/Data/Berts final/{experiment}/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model running on {device}")
    foldFiles = foldFinder(path, fold)
    if subtype == "NORM-PSVT+nonWPW":
        foldFiles = [file for file in foldFiles if "AVRT" not in file]
    elif subtype == "AVNRT-AVRT/concealed":
        foldFiles = [file for file in foldFiles if (("AVRT" in file) or ("concealed" in file) or ("AVNRT" in file))]
    model.load_state_dict(torch.load(weightsPath))
    model.to(device)
    model.eval()
    tempDataset, tempLoader = None, None
    if subtype == "5-class":
        tempDataset = ECGDataset2_0(foldFiles, experiment)
        tempLoader = DataLoader(tempDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
    elif subtype == "NORM-PSVT+nonWPW":
        tempDataset = ECGDataset2_0Binary(foldFiles, experiment, AVRT=False)
        tempLoader = DataLoader(tempDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
    elif subtype == "AVNRT-AVRT+concealed":
        tempDataset = ECGDataset2_0Binary(foldFiles, experiment, AVRT=True)
        tempLoader = DataLoader(tempDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
        
    modelMinusLast = ModelWithoutLastLayer(model)
    createNsaveEmbeddings(model=modelMinusLast, experiment=experiment, modelName=modelName, 
                          batchSize=64, device=device, loader=tempLoader, 
                          subtype=subtype, fold=fold)


def embeddingAccumulator(experiment:str, subtype:str, modelName:str, 
                         fold:int, path:str="/home/tzikos/Desktop/Data/embeddings"):
    trainEmbeddings, valEmbeddings, testEmbeddings = [], None, None
    trainLabels, valLabels, testLabels = [], None, None
    for i in range(10):
        foldEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{i+1}/{modelName}embeddings.npy")
        foldLabels = np.load(f"{path}/{experiment}/{subtype}/fold{i+1}/{modelName}labels.npy")

        if i == fold:
            valEmbeddings = foldEmbeddings
            valLabels = foldLabels
        elif i == (fold+1)%10:
            testEmbeddings = foldEmbeddings
            testLabels = foldLabels
        else:
            trainEmbeddings.append(foldEmbeddings)
            trainLabels.append(foldLabels)
    
    trainEmbeddings = np.concatenate(trainEmbeddings, axis=0)
    trainLabels = np.concatenate(trainLabels, axis=0)

    os.makedirs(f"{path}/{experiment}/{subtype}/fold{fold+1}/train", exist_ok=True)
    os.makedirs(f"{path}/{experiment}/{subtype}/fold{fold+1}/val", exist_ok=True)
    os.makedirs(f"{path}/{experiment}/{subtype}/fold{fold+1}/test", exist_ok=True)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccEmbeddings.npy", trainEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/val/{modelName}AccEmbeddings.npy", valEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccEmbeddings.npy", testEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccLabels.npy", trainLabels)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/val/{modelName}AccLabels.npy", valLabels)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccLabels.npy", testLabels)


def applyLDA(experiment:str, subtype:str, modelName:str,
              fold:int, path:str="/home/tzikos/Desktop/Data/embeddings"):
    trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccEmbeddings.npy")
    valEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/val/{modelName}AccEmbeddings.npy")
    testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccEmbeddings.npy")

    trainLabels = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccLabels.npy")

    lda = LinearDiscriminantAnalysis()
    lda.fit(trainEmbeddings, trainLabels)
    trainEmbeddings = lda.transform(trainEmbeddings)
    valEmbeddings = lda.transform(valEmbeddings)
    testEmbeddings = lda.transform(testEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}LDAEmbeddings.npy", trainEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/val/{modelName}LDAEmbeddings.npy", valEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}LDAEmbeddings.npy", testEmbeddings)



def applySVM(experiment:str, subtype:str,
             modelName:str, useLDA:bool, fold:int, 
             C:float, path:str="/home/tzikos/Desktop/Data/embeddings"):
    if useLDA:
        trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}LDAEmbeddings.npy")
        valEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/val/{modelName}LDAEmbeddings.npy")
        testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}LDAEmbeddings.npy")        
    else:
        trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccEmbeddings.npy")
        valEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/val/{modelName}AccEmbeddings.npy")
        testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccEmbeddings.npy")

    trainLabels = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccLabels.npy")
    valLabels = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/val/{modelName}AccLabels.npy")
    testLabels = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccLabels.npy")

    maxValF1 = 0
    maxTestF1 = 0
    maxTrainF1 = 0
    svm = SVC(C=C)
    svm.fit(trainEmbeddings, trainLabels)
    trainPred = svm.predict(trainEmbeddings)
    valPred = svm.predict(valEmbeddings)
    testPred = svm.predict(testEmbeddings)
    f1Train = f1_score(trainLabels, trainPred, average='macro')
    f1Val = f1_score(valLabels, valPred, average='macro')
    f1Test = f1_score(testLabels, testPred, average='macro')
    print(f"For C={C} the f1 scores are: \n-Train: {f1Train:.2f} \n-Val: {f1Val:.2f} \n-Test: {f1Test:.2f}")

    if f1Val > maxValF1:
        maxTrainF1 = f1Train
        maxValF1 = f1Val
        maxTestF1 = f1Test
        trainCM = confusion_matrix(trainLabels, trainPred)
        valCM = confusion_matrix(valLabels, valPred)
        testCM = confusion_matrix(testLabels, testPred)
    return trainCM, valCM, testCM, maxTrainF1, maxValF1, maxTestF1


def wandbTable(table: np.ndarray, run:wandb.run, classNames:list, modelName:str,
               fold:int, experiment:str, subtype:str, batchSize:int, formattedNow:str,
               saveString:str):
        columns = ['Class'] + [f'Predicted: {className}' for className in classNames]
        CMTable = wandb.Table(columns=columns)
        # Fill the table with data from the confusion matrix
        for i in range(len(table)):
            row = [classNames[i]] + table[i].tolist()
            CMTable.add_data(*row)
        wandb.log({f'{saveString} Confusion Matrix': CMTable}, commit=False)
        # Plot and save confusion matrix
        figTrain = plotNSaveConfusion(table, classNames, f"/Confusion_{modelName}_fold{fold+1}_{experiment}_{subtype}_B{batchSize}_{formattedNow}", f"{saveString}")        
        # Convert Plotly figure to an image and log to W&B
        pio.write_image(figTrain, f"/home/tzikos/Confusions/Confusion_{modelName}_fold{fold+1}_{experiment}_{subtype}_B{batchSize}_{formattedNow}.png", 
                        width=1000, height=1000)
        wandb.log({f"{saveString} Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_{modelName}_fold{fold+1}_{experiment}_{subtype}_B{batchSize}_{formattedNow}.png")})


def hybridML(model: torch.nn.Module, modelName: str,  weightsPath: str, experiment: str,
             batchSize: int, subtype: str, useLDA:bool=True,
             C:float=None):
    trainF1s = []
    valF1s = []
    testF1s = []
    now = datetime.now()
    formattedNow = now.strftime("%d-%m-%y-%H-%M")
    for fold in range(10):
        preprocEmbed(model, modelName, weightsPath, experiment,
                batchSize, subtype, fold)
    for fold in range(10):
        embeddingAccumulator(experiment, subtype, modelName, fold)
        applyLDA(experiment, subtype, modelName, fold)
        trainCM, valCM, testCM, trainF1, valF1, testF1 = applySVM(experiment, subtype,
             modelName, useLDA, fold, C=C)
        run = wandb.init(
                        # set the wandb project where this run will be logged
                        project="KU Thesis - Hybrid ML",
                        # track hyperparameters and run metadata
                        config={
                        "used LDAs": useLDA,
                        "C": C,
                        "fold": fold+1,
                        "batch_size": batchSize,
                        "architecture": weightsPath.split("/")[-1],
                        "experiment": experiment,
                        "subtype": subtype}
                    )

        if subtype == "5-class":
            classNames = ['Normal', 'AVNRT', 'AVRT', 'Concealed', 'EAT']
        elif subtype == "NORM-PSVT+nonWPW":
            classNames = ['Normal', 'PSVT+nonWPW']
        elif subtype == "AVNRT-AVRT+concealed":
            classNames = ['AVNRT', 'AVRT+concealed']
        else:
            raise ValueError("Incorrect subtype, please choose between \n-1) 5-class \n-2) NORM-PSVT/nonWPW \n-3) AVNRT-AVRT/concealed")
        
        wandbTable(trainCM, run, classNames, modelName, fold, experiment, subtype, batchSize, formattedNow, saveString="Training")
        wandbTable(valCM, run, classNames, modelName, fold, experiment, subtype, batchSize, formattedNow, saveString="Validation")
        wandbTable(testCM, run, classNames, modelName, fold, experiment, subtype, batchSize, formattedNow, saveString="Test")
        wandb.log({"Training F1": trainF1, "Validation F1": valF1, "Test F1": testF1})
        trainF1s.append(trainF1)
        valF1s.append(valF1)
        testF1s.append(testF1)
        if fold == 9:
            wandb.run.notes = f"{np.mean(trainF1s*100):.2f}/{np.mean(valF1s*100):.2f}/{np.mean(testF1s*100):.2f}"
        run.finish()
        print("Finished fold ", fold+1)


seedEverything(42)
SUBTYPE = ["5-class", "5-class", "NORM-PSVT+nonWPW", "AVNRT-AVRT+concealed"]
NUMCLASSES = [5, 5, 2, 2]
EXPERIMENT = ["tachy", "pre", "pre", "tachy"]
BATCH_SIZE = 64
WEIGHT_PATHS = ["/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold8_tachy_B64_L1e-06_24-06-24-11-02.pth",
                "/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold7_pre_B64_L5e-06_23-06-24-02-16.pth",
                "/home/tzikos/Desktop/weights/Models/ ECGCNNClassifier_fold7_pre_B64_L3e-07_17-07-24-02-13.pth",
                "/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold7_tachy_B64_L5e-06_11-07-24-08-41.pth"]    
C_LIST = [0.01, 0.1, 1, 10, 100]

for i in range(4):
    for c in C_LIST:
        MODEL = ECGCNNClassifier(NUMCLASSES[i])
        MODELNAME = MODEL.__class__.__name__
        hybridML(MODEL, MODELNAME, WEIGHT_PATHS[i], 
                EXPERIMENT[i], BATCH_SIZE, SUBTYPE[i], useLDA=True, C=c)

