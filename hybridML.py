""" 
This is a hybrid machine learning model pipeline that takes as input the embeddings
of an pretrained model and uses them as input to an LDA model.
The output then is followed by an input to a SVM model.
"""
import torch
from models import foldFinder
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
from models import ECGCNNClassifier, ECGCNNClassifier2
import os
from sklearn.decomposition import KernelPCA

def fileLoader(path:str, model:torch.nn.Module, modelName:str, weightsPath:str, experiment:str, 
               batchSize:int, subtype:str, fold:int, device:torch.device, ignoreMissing:bool=True):
    foldFiles = foldFinder(path, fold, ignoreMissing=ignoreMissing)
    if subtype == "NORM-PSVT+nonWPW":
        foldFiles = [file for file in foldFiles if "AVRT" not in file]
    elif subtype == "AVNRT-AVRT+concealed":
        foldFiles = [file for file in foldFiles if (("AVRT" in file) or ("concealed" in file) or ("AVNRT" in file))]
    #model.load_state_dict(torch.load(weightsPath.replace(r'fold\d', f"fold{fold+1}")))
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
                        subtype=subtype, fold=fold, ignoreMissing=ignoreMissing)


def createNsaveEmbeddings(model, experiment:str, modelName:str,
                          batchSize:int, device:torch.device, loader:DataLoader,
                          subtype:str, fold:int, channels:int=12, timeSteps:int=5000, 
                          ignoreMissing:bool=True):
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
    if ignoreMissing:
        os.makedirs(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/", exist_ok=True)
        np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{modelName}embeddings.npy", allEmbeddings)
        allLabels = allLabels.cpu().numpy()
        #print(f"Labels shape is {allLabels.shape}")
        np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{modelName}labels.npy", allLabels)
    else:
        os.makedirs(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/", exist_ok=True)
        np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{modelName}Allembeddings.npy", allEmbeddings)
        allLabels = allLabels.cpu().numpy()
        #print(f"Labels shape is {allLabels.shape}")
        np.save(f"/home/tzikos/Desktop/Data/embeddings/{experiment}/{subtype}/fold{fold+1}/{modelName}Alllabels.npy", allLabels)


def preprocEmbed(model: torch.nn.Module, modelName: str,  weightsPath: str, experiment: str,
            batchSize: int, subtype:str, fold:int):
    path=f"/home/tzikos/Desktop/Data/Berts final/{experiment}/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model running on {device}")
    fileLoader(path, model, modelName, weightsPath, experiment, batchSize, subtype, fold, device)
    fileLoader(path, model, modelName, weightsPath, experiment, batchSize, subtype, fold, device, ignoreMissing=False)
    


def embeddingAccumulator(experiment:str, subtype:str, modelName:str, 
                         fold:int, path:str="/home/tzikos/Desktop/Data/embeddings",
                         ignoreMissing:bool=True):
    trainEmbeddings, testEmbeddings = [], None
    trainLabels, testLabels = [], None
    for i in range(10):
        foldEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{i+1}/{modelName}embeddings.npy")
        foldLabels = np.load(f"{path}/{experiment}/{subtype}/fold{i+1}/{modelName}labels.npy")

        if i == (fold+1)%10:
            testEmbeddings = foldEmbeddings
            testLabels = foldLabels
        else:
            if ignoreMissing:
                trainEmbeddings.append(foldEmbeddings)
                trainLabels.append(foldLabels)
            elif not ignoreMissing:
                foldAllEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{i+1}/{modelName}Allembeddings.npy")
                foldAllLabels = np.load(f"{path}/{experiment}/{subtype}/fold{i+1}/{modelName}Alllabels.npy")
                trainEmbeddings.append(foldAllEmbeddings)
                trainLabels.append(foldAllLabels)
            else:
                raise ValueError("Please choose between ignoreMissing=True or False")
    
    trainEmbeddings = np.concatenate(trainEmbeddings, axis=0)
    trainLabels = np.concatenate(trainLabels, axis=0)

    os.makedirs(f"{path}/{experiment}/{subtype}/fold{fold+1}/train", exist_ok=True)
    os.makedirs(f"{path}/{experiment}/{subtype}/fold{fold+1}/val", exist_ok=True)
    os.makedirs(f"{path}/{experiment}/{subtype}/fold{fold+1}/test", exist_ok=True)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccEmbeddings.npy", trainEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccEmbeddings.npy", testEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccLabels.npy", trainLabels)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccLabels.npy", testLabels)


def applyPCAnLDA(experiment:str, subtype:str, modelName:str, solver:str,
              fold:int, path:str="/home/tzikos/Desktop/Data/embeddings"):
    trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccEmbeddings.npy")
    testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccEmbeddings.npy")

    trainLabels = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccLabels.npy")


    pca = KernelPCA(n_components=0.95, kernel='rbf')
    allEmbeddings = np.concatenate((trainEmbeddings, testEmbeddings), axis=0)
    pca.fit(allEmbeddings)
    trainEmbeddings = pca.transform(trainEmbeddings)
    testEmbeddings = pca.transform(testEmbeddings)

    lda = LinearDiscriminantAnalysis(solver=solver)
    lda.fit(trainEmbeddings, trainLabels)
    trainEmbeddings = lda.transform(trainEmbeddings)
    testEmbeddings = lda.transform(testEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}LDAEmbeddings.npy", trainEmbeddings)
    np.save(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}LDAEmbeddings.npy", testEmbeddings)


def applySVM(experiment:str, subtype:str,
             modelName:str, useLDA:bool, fold:int, 
             C:float, path:str="/home/tzikos/Desktop/Data/embeddings"):
    if useLDA:
        trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}LDAEmbeddings.npy")
        testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}LDAEmbeddings.npy")        
    else:
        trainEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccEmbeddings.npy")
        testEmbeddings = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccEmbeddings.npy")

    trainLabels = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/train/{modelName}AccLabels.npy")
    testLabels = np.load(f"{path}/{experiment}/{subtype}/fold{fold+1}/test/{modelName}AccLabels.npy")

    svm = SVC(class_weight="balanced", C=C)
    svm.fit(trainEmbeddings, trainLabels)
    trainPred = svm.predict(trainEmbeddings)
    testPred = svm.predict(testEmbeddings)
    f1Train = f1_score(trainLabels, trainPred, average='macro') * 100
    f1Test = f1_score(testLabels, testPred, average='macro') * 100
    print(f"For C={C} the f1 scores are: \n-Train: {f1Train:.2f} \n-Test: {f1Test:.2f}")
    trainCM = confusion_matrix(trainLabels, trainPred)
    testCM = confusion_matrix(testLabels, testPred)
    return trainCM, testCM, f1Train, f1Test


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
             batchSize: int, subtype: str, foldExperiment:int, solver:str,
             useLDA:bool=True, ignoreMissing:bool=True, C:float=None):
    trainF1s = []
    testF1s = []
    now = datetime.now()
    formattedNow = now.strftime("%d-%m-%y-%H-%M")
    for fold in range(10):
        preprocEmbed(model, modelName, weightsPath, experiment,
                batchSize, subtype, fold)
    for fold in range(10):
        if fold == foldExperiment-1:
            embeddingAccumulator(experiment, subtype, modelName, fold, ignoreMissing=ignoreMissing)
            if useLDA:
                applyPCAnLDA(experiment, subtype, modelName, solver, fold)
            trainCM, testCM, trainF1, testF1 = applySVM(experiment, subtype,
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
                            "ignoreMissing": ignoreMissing,
                            "solver": solver,
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
            wandbTable(testCM, run, classNames, modelName, fold, experiment, subtype, batchSize, formattedNow, saveString="Test")
            wandb.log({"Training F1": trainF1, "Test F1": testF1})
            trainF1s.append(trainF1)
            testF1s.append(testF1)
            if fold == 9:
                wandb.run.notes = f"{np.mean(trainF1s*100):.2f}/{np.mean(testF1s*100):.2f}"
            run.finish()
            print("Finished fold ", fold+1)


seedEverything(42)
SUBTYPE = ["5-class", "5-class", "NORM-PSVT+nonWPW", "AVNRT-AVRT+concealed"]
NUMCLASSES = [5, 5, 2, 2]
EXPERIMENT = ["tachy", "pre", "pre", "tachy"]
BATCH_SIZE = 64
WEIGHT_PATHS = ["/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold10_tachy_B64_L5e-05_02-08-24-00-35.pth", # 74.61/61.59
                "/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold2_pre_B64_L1e-06_10-06-24-17-03.pth", # 54.42/47.76    
                "/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold7_pre_B64_L1e-05_01-08-24-01-17.pth", # 79.30/72.04
                "/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold8_tachy_B64_L5e-06_02-08-24-18-06.pth", # 91.75/66.47
]
MODEL_NAME_LIST = [ECGCNNClassifier2(5),
              ECGCNNClassifier(5),
              ECGCNNClassifier2(2),
              ECGCNNClassifier2(2)]

foldExpList = [10, 2, 7, 8]
C_LIST = [0.1, 1, 10]
SOLVER = {#"svd": True
          #"lsqr": True, 
          #"eigen": True, 
          "noLDA": False
          }

for i in range(4):
    for lda in SOLVER.keys():
        for c in C_LIST: 
            MODEL = MODEL_NAME_LIST[i]
            MODELNAME = MODEL.__class__.__name__
            hybridML(MODEL, MODELNAME, WEIGHT_PATHS[i], 
                    EXPERIMENT[i], BATCH_SIZE, SUBTYPE[i], 
                    useLDA=SOLVER[lda], C=c, ignoreMissing=False,
                    foldExperiment=foldExpList[i], solver=lda)

