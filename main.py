from trainingFuncs import train, CVtrain, CVtrainBinary
from preprocessing import originalNBaseNPs, calcMeanSigma, calcClassWeights, foldCreator, processorBert
from preprocPTBXL import readPTB, gaussianNormalizerPTB, dataSplitterPTB
from preprocPTBXL import createMissingLeadsPTB, getGaussianParamsPTB
import numpy as np
import os
from utility import seedEverything
import pickle

# Seed every library for reproducibility
seedEverything(42)

# GLOBALS
CLASSES_BERT = 5
CLASSES_PTB = 5
ROOT_DATA_PATH = '/home/tzikos/Desktop/Data/'
EXPERIMENT = "pre"
WEIGHT_PATH = "/home/tzikos/Desktop/weights/"
PREPROC_PTB = False
PREPROC_BERT = False
PRETRAIN = False
FINETUNE = True
VISUALIZE = False
USE_PRETRAINED = False

# MODEL
BINARY = False
AVNRT_AVRT = False
#MODEL_STR = "CNN2020"
#MODEL_STR = "GatedTransformer"
#MODEL_STR = "CNNAttia"
MODEL_STR = "MLSTMFCN"
#MODEL_STR = "swin"
if MODEL_STR == "swin":
    swin = True
else:
    swin = False


# HYPERPARAMETERS
LEARNING_RATE_PTB = 1e-5
LEARNING_RATE_BERT = 1e-4
BATCH = 64
L1 = None
L2 = 1e-4
A_BERT = 1
A_PTB = 1


typeC = "Diagnostic"
countDict = {"NORM":0, "MI":0, "STTC":0, "CD":0, "HYP":0}
classDict = {"NDT":"STTC", "NST":"STTC", "DIG":"STTC", "LNGQT":"STTC", "NORM":"NORM", 
                    "IMI":"MI", "ASMI":"MI", "LVH":"HYP", "LAFB":"CD", "ISC_":"STTC", 
                    "IRBBB":"CD", "1AVB":"CD", "IVCD":"CD", "ISCAL":"STTC", "CRBBB":"CD",
                    "CLBBB":"CD", "ILMI":"MI", "LAO/LAE":"HYP", "AMI":"MI", "ALMI":"MI",
                    "ISCIN":"STTC", "INJAS":"MI", "LMI":"MI", "ISCIL":"STTC", "LPFB":"CD", 
                    "ISCAS":"STTC", "INJAL":"MI", "ISCLA":"STTC", "RVH":"HYP", "ANEUR":"STTC", 
                    "RAO/RAE":"HYP", "EL":"STTC", "WPW":"CD", "ILBBB":"CD", "IPLMI":"MI",
                    "ISCAN":"STTC", "IPMI":"MI", "SEHYP":"HYP", "INJIN":"MI", "INJLA":"MI",
                    "PMI":"MI", "3AVB":"CD", "INJIL":"MI", "2AVB":"CD"}


if PREPROC_PTB:
    metadataPath="/home/tzikos/Desktop/Data/PTBXL/ptbxl_database.csv"
    dataPath = "/home/tzikos/Desktop/Data/PTBXL/records500/"
    savePath1 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} Denoised and Orig/"
    for key in countDict.keys():
        os.makedirs(f"{savePath1}{key}", exist_ok=True)
    savePath2 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} Normalized/"
    for key in countDict.keys():
        os.makedirs(f"{savePath2}train", exist_ok=True)
    savePath3 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} torch/"
    os.makedirs(f"{savePath3}train", exist_ok=True)
    os.makedirs(f"{savePath3}val", exist_ok=True)
    print("Reading and transferring PTB data")
    countNorm = readPTB(countDict, classDict, metadataPath, dataPath, savePath1)
    print("Starting PTB data splitting")
    trainList, valList = dataSplitterPTB(savePath1, list(countDict.keys()), countNorm=4068)
    print(len(trainList))
    print(len(valList))
    with open(f'{ROOT_DATA_PATH}/PTBtrain.pkl', 'wb') as f:
        pickle.dump(trainList, f)
    with open(f'{ROOT_DATA_PATH}/PTBval.pkl', 'wb') as f:
       pickle.dump(valList, f)
    print("Calculating gaussian parameters for normalizing")
    with open(f'{ROOT_DATA_PATH}/PTBtrain.pkl', 'rb') as f:
        trainList = pickle.load(f)
    getGaussianParamsPTB(trainList)
    with open(f'{ROOT_DATA_PATH}/PTBval.pkl', 'rb') as f:
        valList = pickle.load(f)
    print("Normalizing PTB data")
    print(f"The classes are {list(countDict.keys())}")
    for i in list(countDict.keys()):
        gaussianNormalizerPTB(trainList, valList, savePath2, savePath3)
        print(f"Done with {i}")
    #shutil.rmtree(savePath1)
    print("Creating missing leads for PTB data")
    createMissingLeadsPTB(savePath2, savePath3)
    print("Done PTB preprocessing")
    #shutil.rmtree(savePath2)


expList = [f'normal {EXPERIMENT}', f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']
if PREPROC_BERT:
    print("Creating original numpys")
    originalNBaseNPs(expList)
    print("Creating folds")
    folds = foldCreator(expList)
    print("Calculating mean and sigma from normal data")
    calcMeanSigma(folds, EXPERIMENT)
    print("Processing folds")
    processorBert(folds, expList)
    print("Calculating class weights")
    calcClassWeights(expList)
    print("Done Berts preprocessing")

if PRETRAIN:
    # Datasets
    ptbList = [f'NORM ptb', f'MI ptb', f'STTC ptb', f'CD ptb', f'HYP ptb']
    classWeights = np.load(WEIGHT_PATH + f"PTBweights{CLASSES_PTB}.npy")
    classWeights = A_PTB * classWeights
    print(f"Class weights are {classWeights}")
    train(modelStr=MODEL_STR, learningRate=LEARNING_RATE_PTB, classWeights=classWeights, expList=list(countDict.keys()),
          batchSize=BATCH, modelWeightPath=f"{WEIGHT_PATH}Models", L1=L1, L2=L2)
    

if FINETUNE:
    calcClassWeights(expList)
    if BINARY:
        expList = [f'normal {EXPERIMENT}', f'PSVT {EXPERIMENT}']
        classWeights = np.load(f'{WEIGHT_PATH}{EXPERIMENT}ClassWeightsBinary.npy')
    elif AVNRT_AVRT:
        expList = [f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}']
        classWeights = np.load(f'{WEIGHT_PATH}{EXPERIMENT}ClassWeightsAVRT.npy')
    else:
        classWeights = np.load(f'{WEIGHT_PATH}{EXPERIMENT}ClassWeights.npy')
    classWeight = A_BERT * classWeights
    print(classWeights)
    if BINARY or AVNRT_AVRT:
        CVtrainBinary(modelStr=MODEL_STR, learningRate=LEARNING_RATE_BERT, epochs=1000, classWeights=classWeights,
                earlyStopPatience=12, reduceLRPatience=5, expList=expList, 
                dataset=EXPERIMENT, batchSize=BATCH, L1=L1, L2=L2, usePretrained=USE_PRETRAINED, 
                modelWeightPath=f"{WEIGHT_PATH}Models", scaler=A_BERT, swin=swin, AVRT=AVNRT_AVRT)
    else:
        CVtrain(modelStr=MODEL_STR, learningRate=LEARNING_RATE_BERT, epochs=1000, classWeights=classWeights,
                earlyStopPatience=12, reduceLRPatience=5, expList=expList, 
                dataset=EXPERIMENT, batchSize=BATCH, L1=L1, L2=L2, usePretrained=USE_PRETRAINED, 
                modelWeightPath=f"{WEIGHT_PATH}Models", scaler=A_BERT, swin=swin)
