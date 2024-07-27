""" 
This is a hybrid machine learning model pipeline that takes as input the embeddings
of an pretrained model and uses them as input to an LDA model.
The output then is followed by an input to a SVM model.
"""
import torch
import time
from models import lengthFinder, lengthFinderBinary, foldFinder
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

def createNsaveEmbeddings(model, experiment:str, modelName:str,
                          batchSize:int, device:torch.device, loader:DataLoader,
                          subtype:str, fold:int):
    embeddingSize = model(torch.zeros(batchSize, 12, 5000).to(device))
    allEmbeddings = torch.empty(embeddingSize).to(device)
    allLabels = torch.empty(0).to(device)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            allEmbeddings = torch.cat((allEmbeddings, outputs))
            allLabels = torch.cat((allLabels, labels))
    allEmbeddings = allEmbeddings.cpu().numpy()
    np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{modelName}embeddings.npy", allEmbeddings)
    allLabels = allLabels.cpu().numpy()
    np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{modelName}labels.npy", allLabels)

def preprocEmbed(model: torch.nn.Module, modelName: str,  weightsPath: str, experiment: str,
            batchSize: int, subtype:str, fold:int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model running on {device}")
    foldFiles = foldFinder(experiment, fold)
    if subtype == "5-class":
        numClasses = 5
    elif subtype == "NORM-PSVT/nonWPW":
        numClasses = 2
        foldFiles = [file for file in foldFiles if "AVRT" not in file]
    elif subtype == "AVNRT-AVRT/concealed":
        numClasses = 2
        foldFiles = [file for file in foldFiles if (("AVRT" in file) or ("concealed" in file) or ("AVNRT" in file))]
    else:
        raise ValueError("Incorrect subtype, please choose between \n-1) 5-class \n-2) NORM-PSVT/nonWPW \n-3) AVNRT-AVRT/concealed")
    model = model(numClasses=numClasses)
    model.load_state_dict(torch.load(weightsPath))
    model.to(device)
    model.eval()
    if subtype == "5-class":
        tempDataset = ECGDataset2_0(foldFiles, experiment)
        tempLoader = DataLoader(tempDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
    elif subtype == "NORM-PSVT/nonWPW":
        tempDataset = ECGDataset2_0Binary(foldFiles, experiment, AVRT=False)
        tempLoader = DataLoader(tempDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
    elif subtype == "AVNRT-AVRT/concealed":
        tempDataset = ECGDataset2_0Binary(foldFiles, experiment, AVRT=True)
        tempLoader = DataLoader(tempDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
        
    modelMinusLast = ModelWithoutLastLayer(model)
    createNsaveEmbeddings(modelMinusLast, "train", experiment, modelName, weightsPath, batchSize, device, tempLoader, fold)


def embeddingAccumulator(experiment:str, subtype:str, modelName:str, 
                         fold:int, path:str="/home/tzikos/Desktop/Data/embeddings"):
    trainEmbeddings, valEmbeddings, testEmbeddings = [], None, None
    trainLabels, valLabels, testLabels = [], None, None
    for i in range(1, 10):
        foldEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}embeddings.npy")
        foldLabels = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}labels.npy")
        if i == fold:
            valEmbeddings = foldEmbeddings
            valLabels = foldLabels
        elif i == fold+1:
            testEmbeddings = foldEmbeddings
            testLabels = foldLabels
        else:
            trainEmbeddings.append(foldEmbeddings)
            trainLabels.append(foldLabels)
    
    trainEmbeddings = np.concatenate(trainEmbeddings, axis=0)
    trainLabels = np.concatenate(trainLabels, axis=0)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}AccEmbeddings.npy", trainEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}AccEmbeddings.npy", valEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}AccEmbeddings.npy", testEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}AccLabels.npy", trainLabels)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}AccLabels.npy", valLabels)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}AccLabels.npy", testLabels)

def applyLDA(experiment:str, subtype:str, modelName:str,
              fold:int, path:str="/home/tzikos/Desktop/Data/embeddings"):
    trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}AccEmbeddings.npy")
    valEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}AccEmbeddings.npy")
    testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}AccEmbeddings.npy")

    trainLabels = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}AccLabels.npy")
    valLabels = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}AccLabels.npy")
    testLabels = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}AccLabels.npy")

    lda = LinearDiscriminantAnalysis()
    lda.fit(trainEmbeddings, trainLabels)
    trainEmbeddings = lda.transform(trainEmbeddings)
    valEmbeddings = lda.transform(valEmbeddings)
    testEmbeddings = lda.transform(testEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}LDAEmbeddings.npy", trainEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}LDAEmbeddings.npy", valEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}LDAEmbeddings.npy", testEmbeddings)



def applySVM(experiment:str, subtype:str,
             modelName:str, useLDA:bool, fold:int, 
             C:float=None, path:str="/home/tzikos/Desktop/Data/embeddings"):
    if useLDA:
        trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}LDAEmbeddings.npy")
        valEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}LDAEmbeddings.npy")
        testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}LDAEmbeddings.npy")        
    else:
        trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}AccEmbeddings.npy")
        valEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}AccEmbeddings.npy")
        testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}AccEmbeddings.npy")

    trainLabels = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}labels.npy")
    valLabels = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}labels.npy")
    testLabels = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}labels.npy")

    if C is None:
        CList = [0.01, 0.1, 1, 10, 100]
    else:
        CList = [C]

    maxValF1 = 0
    bestC = 0
    for c in CList:
        svm = SVC(C=c)
        svm.fit(trainEmbeddings, trainLabels)
        trainPred = svm.predict(trainEmbeddings)
        valPred = svm.predict(valEmbeddings)
        testPred = svm.predict(testEmbeddings)
        f1Train = f1_score(trainLabels, trainPred, average='macro')
        f1Val = f1_score(valLabels, valPred, average='macro')
        f1Test = f1_score(testLabels, testPred, average='macro')
        print(f"For C={c} the f1 scores are: \n-Train: {f1Train:.2f} \n-Val: {f1Val:.2f} \n-Test: {f1Test:.2f}")

        if f1Val > maxValF1:
            maxValF1 = f1Val
            bestC = c
            trainCM = confusion_matrix(trainLabels, trainPred)
            valCM = confusion_matrix(valLabels, valPred)
            testCM = confusion_matrix(testLabels, testPred)
    if len(CList) == 1:
        return trainCM, valCM, testCM


def wandbTable(table: np.ndarray, run:wandb.run, classNames:list, modelName:str,
               fold:int, experiment:str, subtype:str, batchSize:int, formattedNow:str):
        columns = ['Class'] + [f'Predicted: {className}' for className in classNames]
        CMTable = wandb.Table(columns=columns)
        # Fill the table with data from the confusion matrix
        for i in range(len(table)):
            row = [classNames[i]] + table[i].tolist()
            CMTable.add_data(*row)
        wandb.log({'Training Confusion Matrix': CMTable}, commit=False)
        # Plot and save confusion matrix
        figTrain = plotNSaveConfusion(table, classNames, f"/Confusion_train_{modelName}_fold{fold+1}_{experiment}_{subtype}_B{batchSize}_{formattedNow}", "Train")        
        # Convert Plotly figure to an image and log to W&B
        pio.write_image(figTrain, f"/home/tzikos/Confusions/Confusion_train_{modelName}_fold{fold+1}_{experiment}_{subtype}_B{batchSize}_{formattedNow}.png", 
                        width=1000, height=1000)
        wandb.log({"Training Confusion Matrix Heatmap": wandb.Image(f"/home/tzikos/Confusions/Confusion_train_{modelName}_fold{fold+1}_{experiment}_{subtype}_B{batchSize}_{formattedNow}.png")})

def hybridML(model: torch.nn.Module, modelName: str,  weightsPath: str, experiment: str,
             batchSize: int, subtype: str, notes: str=None, useLDA:bool=True,
             C:float=None):
    now = datetime.now()
    formattedNow = now.strftime("%d-%m-%y-%H-%M")
    for fold in range(10):
        preprocEmbed(model, modelName, weightsPath, experiment,
                batchSize, subtype, fold)
        embeddingAccumulator(experiment, subtype, modelName, fold)
        applyLDA(experiment, subtype, modelName, fold)
        trainCM, valCM, testCM = applySVM(experiment, subtype,
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
        wandb.run.notes = notes

        if subtype == "5-class":
            classNames = ['AVNRT', 'AVRT', 'EAT', 'Normal', 'Concealed']
        elif subtype == "NORM-PSVT/nonWPW":
            classNames = ['Normal', 'PSVT/nonWPW']
        elif subtype == "AVNRT-AVRT/concealed":
            classNames = ['AVNRT', 'AVRT/concealed']
        else:
            raise ValueError("Incorrect subtype, please choose between \n-1) 5-class \n-2) NORM-PSVT/nonWPW \n-3) AVNRT-AVRT/concealed")
        
        wandbTable(trainCM, run, classNames, modelName, fold, experiment, subtype, batchSize, formattedNow)
        wandbTable(valCM, run, classNames, modelName, fold, experiment, subtype, batchSize, formattedNow)
        wandbTable(testCM, run, classNames, modelName, fold, experiment, subtype, batchSize, formattedNow)
        run.finish()




seedEverything(42)

SUBTYPE = "5-class"
NUMCLASSES = 5
EXPERIMENT = "pre"
BATCH_SIZE = 64
MODEL = CNN2023Attia(5)
MODELNAME = MODEL.__class__.__name__

hybridML(MODEL, MODELNAME, "/home/tzikos/Desktop/weights/CNN2023Attia_fold9_pre_B64_L1e-05_26-07-24-03-27.pth", EXPERIMENT, BATCH_SIZE, SUBTYPE, useLDA=True, C=1)

