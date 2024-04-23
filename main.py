from models import PTBDataset
from preprocessing import originalNBaseNPs, calcMeanSigma, calcClassWeights, foldCreator, processorBert
from preprocPTBXL import readPTB, gaussianNormalizerPTB, dataSplitterPTB, createMissingLeadsPTB
import numpy as np
import os
from utility import seedEverything

seedEverything(42)

# HYPERPARAMETERS
LEARNING_RATE_PTB = 0.001
LEARNING_RATE_BERT = 1e-5
BATCH = 64
L1 = None
L2 = 1
A_BERT = 1
A_PTB = 1

# GLOBALS
CLASSES_BERT = 5
CLASSES_PTB = 12
ROOT_DATA_PATH = '/home/tzikos/Desktop/Data/'
EXPERIMENT = "tachy"
WEIGHT_PATH = "/home/tzikos/Desktop/weights/"
PREPROC_PTB = False 
PREPROC_BERT = False
PRETRAIN = False
FINETUNE = True
VISUALIZE = False
USE_PRETRAINED = False
MODEL_STR = "CNN2020"
#MODEL_STR = "GatedTransformer"
#MODEL_STR = "CNNAttia"
#MODEL_STR = "MLSTMFCN"
#MODEL_STR = "swin"
if MODEL_STR == "swin":
    swin = True
else:
    swin = False


typeC = "Diagnostic"
countDict = {"NORM":0, "MI":0, "STTC":0, "CD":0, "HYP":0}
classDict = {"NDT":"STTC", "NST":"STTC", "DIG":"STTC", "LNGQT":"STTC", "NORM":"NORM", 
                    "IMI":"MI", "ASMI":"MI", "LVH":"HYP", "LAFB":"CD", "ISC_":"STTC", 
                    "IRBBB":"CD", "1AVB":"CD", "IVCD":"CD", "ISCAL":"STTC", "CRBBB":"CD",
                    "CLBBB":"CD", "ILMI":"MI", "LAO/LAE":"HYP", "AMI":"MI", "ALMI":"MI",
                    "ISCIN":"STTC", "INJAS":"MI", "LMI":"MI", "ISCIL":"STTC", "LPFB":"CD", 
                    "ISCAS":"STTC", "INJAL":"MI", "ISCLA":"STTC", "RVH":"HYP", "ANEUR":"STTC", 
                    "RAO/RAE":"HYP", "EL":"STTC"    , "WPW":"CD", "ILBBB":"CD", "IPLMI":"MI",
                    "ISCAN":"STTC", "IPMI":"MI", "SEHYP":"HYP", "INJIN":"MI", "INJLA":"MI",
                    "PMI":"MI", "3AVB":"CD", "INJIL":"MI", "2AVB":"CD"}


if PREPROC_PTB:
    metadataPath="/home/tzikos/Desktop/Data/PTBXL/ptbxl_database.csv"
    dataPath = "/home/tzikos/Desktop/Data/PTBXL/records500/"
    savePath1 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} Downsampled/"
    for key in countDict.keys():
        os.makedirs(f"{savePath1}{key}", exist_ok=True)
    savePath2 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} Denoised/"
    for key in countDict.keys():
        os.makedirs(f"{savePath2}{key}", exist_ok=True)
    savePath3 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} Gaussian/"
    for key in countDict.keys():
        os.makedirs(f"{savePath3}{key}", exist_ok=True)
    savePath4 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} torch/"
    os.makedirs(f"{savePath4}train", exist_ok=True)
    os.makedirs(f"{savePath4}val", exist_ok=True)
    tempPath = readPTB(countDict, classDict, metadataPath, dataPath, savePath1, savePath2)
    trainList, valList = dataSplitterPTB(tempPath, list(classDict.values()))
    gaussianNormalizerPTB(trainList, savePath2, CLASSES_PTB, savePath3)
    createMissingLeadsPTB(savePath3, countDict, savePath4)

expList = [f'normal {EXPERIMENT}', f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']

if PREPROC_BERT:
    print("Creating original numpys")
    originalNBaseNPs(expList)
    print("Calculating mean and sigma from normal data")
    calcMeanSigma(f"{ROOT_DATA_PATH}/Berts orig/{EXPERIMENT}", EXPERIMENT)
    print("Creating folds")
    folds = foldCreator(expList)
    print("Processing folds")
    processorBert(folds, expList)
    print("Calculating class weights")
    calcClassWeights(expList)
    print("Done Berts preprocessing")


if PRETRAIN:
    # Datasets
    classWeights = np.load(WEIGHT_PATH + f"PTBweights{CLASSES_PTB}.npy")
    classWeights = A_PTB * classWeights
    #print(classWeights.shape)
    #print(classWeights)
    trainDataset = PTBDataset(f"{ROOT_DATA_PATH}/PTBXL {typeC} torch/train/", CLASSES_PTB, classWeights, countDict)
    trainLoader = DataLoader(trainDataset, batch_size=BATCH, shuffle=True, num_workers=8)
    valDataset = PTBDataset(f"{ROOT_DATA_PATH}/PTBXL {typeC} torch/val/", CLASSES_PTB, classWeights, countDict)
    valLoader = DataLoader(valDataset, batch_size=BATCH, shuffle=True, num_workers=8)
    filePath = train(model, trainLoader, valLoader, CLASSES_PTB, LEARNING_RATE_PTB,
          EPOCHS, classWeights, EARLY_PAT, REDLR_PAT, device, list(countDict.keys()),
          dataset="PTBXL", batchSize=BATCH, lr=LEARNING_RATE_PTB, L1=L1, L2=L2)
    

if FINETUNE:
    calcClassWeights(expList)
    classWeights = np.load(f'{WEIGHT_PATH}{EXPERIMENT}ClassWeights.npy')
    print(classWeights)
    classWeight = A_BERT * classWeights
    CVtrain(modelStr=MODEL_STR, learningRate=LEARNING_RATE_BERT, epochs=1000, classWeights=classWeights,
            earlyStopPatience=12, reduceLRPatience=5, expList=expList, 
            dataset=EXPERIMENT, batchSize=BATCH, L1=L1, L2=L2, usePretrained=USE_PRETRAINED, 
            modelWeightPath=f"{WEIGHT_PATH}Models", scaler=A_BERT, swin=swin)
