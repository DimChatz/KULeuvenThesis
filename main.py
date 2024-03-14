from models import ECGDataset, PTBDataset, train, test, ECGCNNClassifier, ECGSimpleClassifier
from torch.utils.data import DataLoader
from preprocessing import appendStratifiedFiles, downsamplerNoiseRemover, gaussianCalcBert, gaussianNormalizerMissingLeadCreatorBert
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
L1 = False
L2 = False

# Stable
EPOCHS = 1000
EARLY_PAT = 12
REDLR_PAT = 5

# GLOBALS
CLASSES_BERT = 5
CLASSES_PTB = 5
ROOT_DATA_PATH = '/home/tzikos/Desktop/Data'
EXPERIMENT = "pre"
WEIGHT_PATH = "/home/tzikos/Desktop/weights/"
PREPROC_PTB = False
PREPROC_BERT = True
PRETRAIN = False
FINETUNE = False
TEST = False
VISUALIZE = False



if PREPROC_PTB:
    if CLASSES_PTB == 5:
        readPTB_5()
    gaussianNormalizerPTB()
    trainList, valList = dataSplitterPTB()
    createMissingLeadsPTB(trainList, "train")
    createMissingLeadsPTB(valList, "val")

expList = [f'normal {EXPERIMENT}', f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']
PTBList = ["NORM", "MI", "STTC", "CD", "HYP"]

if PREPROC_BERT:
    #trainFiles, valFiles, testFiles = appendStratifiedFiles(f"{ROOT_DATA_PATH}/Berts/", expList)
    #downsamplerNoiseRemover(trainFiles, f"{ROOT_DATA_PATH}/Berts downsapled/{EXPERIMENT}/train", 
    #                        f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/train")
    #downsamplerNoiseRemover(valFiles, f"{ROOT_DATA_PATH}/Berts downsapled/{EXPERIMENT}/val", 
    #                        f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/val")    
    #downsamplerNoiseRemover(testFiles, f"{ROOT_DATA_PATH}/Berts downsapled/{EXPERIMENT}/test", 
    #                        f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/test")
    #print("End of downsampling and noise removal")
    mean, sigma = gaussianCalcBert(f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/train")
    print(mean)
    print(mean.shape)
    print(sigma)
    print(sigma.shape)
    gaussianNormalizerMissingLeadCreatorBert(f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/train", 
                                             f"{ROOT_DATA_PATH}/Berts gaussian/{EXPERIMENT}/train", 
                                             f"{ROOT_DATA_PATH}/Berts torch/{EXPERIMENT}/train",
                                             mean, sigma) 
    gaussianNormalizerMissingLeadCreatorBert(f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/val", 
                                             f"{ROOT_DATA_PATH}/Berts gaussian/{EXPERIMENT}/val", 
                                             f"{ROOT_DATA_PATH}/Berts torch/{EXPERIMENT}/val",
                                             mean, sigma) 
    gaussianNormalizerMissingLeadCreatorBert(f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/test", 
                                             f"{ROOT_DATA_PATH}/Berts gaussian/{EXPERIMENT}/test", 
                                             f"{ROOT_DATA_PATH}/Berts torch/{EXPERIMENT}/test",
                                             mean, sigma)
    print("End of Berts preprocessing")
    

model = ECGCNNClassifier(CLASSES_PTB)
'''
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
    test(model, testLoader, CLASSES_BERT, device, filePath, expList)'''