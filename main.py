from models import ECGDataset, PTBDataset, train, test, ECGCNNClassifier, ECGSimpleClassifier
from torch.utils.data import DataLoader
from preprocessing import gaussianNormalizer, preprocessAll
from utility import calcWeights, classInstanceCalc
from visualizer import preprocVis
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from preprocPTBXL import readPTB_5, gaussianNormalizerPTB, dataSplitterPTB, createMissingLeadsPTB
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# HYPERPARAMETERS
LEARNING_RATE_PTB = 0.0000003
LEARNING_RATE_BERT = 0.0000003
BATCH = 64

# Stable
EPOCHS = 1000
EARLY_PAT = 12
REDLR_PAT = 5

# GLOBALS
CLASSES_BERT = 5
CLASSES_PTB = 5
EXPERIMENT = "pre"
PTB_PATH = "/home/tzikos/Desktop/Data/PTBXL torch/"
ROOT_PATH = '/home/tzikos/Desktop/Data/Berts/'
DIR_PATH = f"/home/tzikos/Desktop/Data/Berts torch/{EXPERIMENT}"
WEIGHT_PATH = "/home/tzikos/Desktop/weights/"
PREPROC_PTB = False
PRETRAIN = False
PREPROC_BERT = True
CALC_WEIGHTS_BERT = True
FINETUNE = False
TEST = False



if PREPROC_PTB:
    if CLASSES_PTB == 5:
        readPTB_5()
    gaussianNormalizerPTB()
    trainList, valList = dataSplitterPTB()
    createMissingLeadsPTB(trainList, "train")
    createMissingLeadsPTB(valList, "val")

expList = [f'normal {EXPERIMENT}, "AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']
PTBList = ["NORM", "MI", "STTC", "CD", "HYP"]

if PREPROC_BERT:
    mean, sigma = gaussianNormalizer(ROOT_PATH, segment = f'{EXPERIMENT}', experiment=EXPERIMENT)
    print(f"mean is {mean}, and sigma is {sigma}")
    preprocessAll(ROOT_PATH, expList, mean, sigma)
    preprocVis("/home/tzikos/Desktop/Data/Berts/AVNRT/AVNRT tachy/AVNRT tachy/75E6EFE9-4EC8-4B3A-9E89-5004E6A326F2_2AX1ZWAX3DRF_20220426_2.xlsx",
               mean[0], sigma[0])
    
model = ECGCNNClassifier(CLASSES_PTB)

if PRETRAIN:
    # Datasets
    classWeights = np.load(WEIGHT_PATH + "PTBweights.npy")
    print(classWeights.shape)
    print(classWeights)
    trainDataset = PTBDataset(f"{PTB_PATH}train/", CLASSES_PTB, classWeights)
    trainLoader = DataLoader(trainDataset, batch_size=BATCH, shuffle=True, num_workers=8)
    valDataset = PTBDataset(f"{PTB_PATH}val/", CLASSES_PTB, classWeights)
    valLoader = DataLoader(valDataset, batch_size=BATCH, shuffle=True, num_workers=8)
    filePath = train(model, trainLoader, valLoader, CLASSES_PTB, LEARNING_RATE_PTB,
          EPOCHS, classWeights, EARLY_PAT, REDLR_PAT, device, PTBList,
          dataset="PTBXL", batchSize=BATCH, lr=LEARNING_RATE_PTB)



if CALC_WEIGHTS_BERT:
     classWeights = classInstanceCalc(ROOT_PATH, WEIGHT_PATH, expList, EXPERIMENT)
classWeights = calcWeights(WEIGHT_PATH, EXPERIMENT, CLASSES_BERT)

if FINETUNE:
    # Datasets
    trainDataset = ECGDataset(f"{DIR_PATH}/train/", EXPERIMENT, CLASSES_BERT, classWeights)
    stratTrainSampler = WeightedRandomSampler(weights=trainDataset, num_samples=len(trainDataset) )
    trainLoader = DataLoader(trainDataset, batch_size=BATCH, shuffle=True, num_workers=8)
    valDataset = ECGDataset(f"{DIR_PATH}/val/", EXPERIMENT, CLASSES_BERT, classWeights)
    stratValSampler = WeightedRandomSampler(weights=valDataset, num_samples=len(valDataset) )
    valLoader = DataLoader(valDataset, batch_size=BATCH, shuffle=True, num_workers=8)
    filePath = train(model, trainLoader, valLoader, CLASSES_BERT, LEARNING_RATE_BERT,
          EPOCHS, classWeights, EARLY_PAT, REDLR_PAT, device, expList,
          dataset=EXPERIMENT, batchSize=BATCH, lr=LEARNING_RATE_BERT)
    
if TEST:
    if FINETUNE:
        filePath = filePath
    else:
        filePath = f"/home/tzikos/Desktop/weights/ECGCNNClassifier_07-03-24-12-13.pth"
    testDataset = ECGDataset(f"{DIR_PATH}/test/", EXPERIMENT, CLASSES_BERT, classWeights)
    testLoader = DataLoader(testDataset, batch_size=BATCH)
    test(model, testLoader, CLASSES_BERT, device, filePath, expList)