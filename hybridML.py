""" 
This is a hybrid machine learning model pipeline that takes as input the embeddings
of an pretrained model and uses them as input to an LDA model.
The output then is followed by an input to a SVM model.
"""
import torch
import time
from models import lengthFinder, lengthFinderBinary
import re
import wandb
from models import ECGDataset2_0, ECGDataset2_0Binary
from torch.utils.data import DataLoader
from utility import ModelWithoutLastLayer
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def createNsaveEmbeddings(model, experiment:str, modelName:str,
                          batchSize:int, device:torch.device, loader:DataLoader,
                          subtype:str, fold:int, dataModality:str):
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
    np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{dataModality}/{modelName}embeddings.npy", allEmbeddings)
    allLabels = allLabels.cpu().numpy()
    np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{dataModality}/{modelName}labels.npy", allLabels)

def embed(modelDict: dict, modelName: str,  weightsPath: str, experiment: str,
            batchSize: int, subtype:str, fold:int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model running on {device}")
    if subtype == "5-class":
        trainFileList, valFileList, testFileList = lengthFinder(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/", fold)
        numClasses = 5
    elif subtype == "NORM-PSVT/nonWPW":
        trainFileList, valFileList, testFileList = lengthFinder(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/", fold, norm_psvt=True)
        numClasses = 2
    elif subtype == "AVNRT-AVRT/concealed":
        trainFileList, valFileList, testFileList = lengthFinderBinary(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/", fold, weightsPath)
        numClasses = 2
    else:
        raise ValueError("Incorrect subtype, please choose between \n-1) 5-class \n-2) NORMPSVT/nonWPW \n-3) AVNRT-AVRT/concealed")
    model = modelDict[modelName](numClasses=numClasses)
    model.load_state_dict(torch.load(weightsPath))
    model.to(device)
    model.eval()
    if subtype == "5-class":
        trainDataset = ECGDataset2_0(trainFileList, experiment)
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
        valDataset = ECGDataset2_0(valFileList, experiment)
        valLoader = DataLoader(valDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
        testDataset = ECGDataset2_0(testFileList, experiment)
        testLoader = DataLoader(testDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
    else:
        trainDataset = ECGDataset2_0Binary(trainFileList, experiment, AVRT=(subtype=="AVNRT-AVRT/concealed"))
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
        valDataset = ECGDataset2_0Binary(valFileList, experiment, AVRT=(subtype=="AVNRT-AVRT/concealed"))
        valLoader = DataLoader(valDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
        testDataset = ECGDataset2_0Binary(testFileList, experiment, AVRT=(subtype=="AVNRT-AVRT/concealed"))
        testLoader = DataLoader(testDataset, batch_size=batchSize, 
                            shuffle=True, 
                            num_workers=8)
        
    modelMinusLast = ModelWithoutLastLayer(model)
    createNsaveEmbeddings(modelMinusLast, "train", experiment, modelName, weightsPath, batchSize, device, trainLoader, fold)
    createNsaveEmbeddings(modelMinusLast, "val", experiment, modelName, weightsPath, batchSize, device, valLoader, fold)
    createNsaveEmbeddings(modelMinusLast, "test", experiment, modelName, weightsPath, batchSize, device, testLoader, fold)


def applyLDA(path:str, experiment:str, subtype:str, modelName:str, fold:int,
             dataModality:str):
    trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}embeddings.npy")
    valEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}embeddings.npy")
    testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}embeddings.npy")

    trainLabels = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}labels.npy")

    lda = LinearDiscriminantAnalysis()
    lda.fit(trainEmbeddings, trainLabels)
    trainEmbeddings = lda.transform(trainEmbeddings)
    valEmbeddings = lda.transform(valEmbeddings)
    testEmbeddings = lda.transform(testEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}LDAEmbeddings.npy", trainEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}LDAEmbeddings.npy", valEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}LDAEmbeddings.npy", testEmbeddings)



def applySVM(path:str, experiment:str, subtype:str,
             modelName:str, useLDA:bool, fold:int, C:float=None):
    if useLDA:
        trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}LDAEmbeddings.npy")
        valEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}LDAEmbeddings.npy")
        testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}LDAEmbeddings.npy")        
    else:
        trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/train/{modelName}embeddings.npy")
        valEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/val/{modelName}embeddings.npy")
        testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/{fold+1}/test/{modelName}embeddings.npy")

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
    return trainCM, valCM, testCM


def hybridML(modelDict: dict, modelName: str,  weightsPath: str, experiment: str):
    print("Hello")