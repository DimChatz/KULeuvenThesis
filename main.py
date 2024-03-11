from models import ECGDataset, train, test, ECGCNNClassifier, ECGSimpleClassifier
from torch.utils.data import DataLoader
from preprocessing import gaussianNormalizer, preprocessAll
from utility import calcWeights, classInstanceCalc
from visualizer import preprocVis, Vis
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from preprocPTBXL import preprocPTBXL, getGaussianParamsPTB, gaussianNormalizerPTB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# HYPERPARAMETERS
LEARNING_RATE = 0.0000005
BATCH = 64

# Stable
EPOCHS = 1000
EARLY_PAT = 12
REDLR_PAT = 5

# GLOBALS
CLASSES_BERT = 4
CLASSES_PTB = 5
PREPROC_PTB = False
FINETUNE = True
EXPERIMENT = "pre"
PTB_PATH = "/home/tzikos/Desktop/Data/PTBXLProcessed/"
DIR_PATH = f"/home/tzikos/Desktop/Data/Berts torch/{EXPERIMENT}"
WEIGHT_PATH = "/home/tzikos/Desktop/weights/"
PREPROC_BERT = False
CALC_WEIGHTS_BERT = False
CALC_WEIGHTS_PTB = False
FINETUNING = True
PRETRAIN = False
TEST = False
ROOT_PATH = '/home/tzikos/Desktop/Data/Berts/'


if PREPROC_PTB:
    preprocPTBXL(metadataPath="/home/tzikos/Desktop/Data/PTBXL/ptbxl_database.csv", 
                 dataPath = "/home/tzikos/Desktop/Data/PTBXL/records500/", 
                 savePath = "/home/tzikos/Desktop/Data/PTBXLProcessed/") 
    mean, sigma = getGaussianParamsPTB(PTB_PATH)
    gaussianNormalizerPTB(PTB_PATH)
#Vis("/home/tzikos/Desktop/Data/PTBXLProcessed/NORM/1.npy")


expList = [f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']

if PREPROC_BERT:
    mean, sigma = gaussianNormalizer(ROOT_PATH, segment = f'{EXPERIMENT}')
    print(f"mean is {mean}, and sigma is {sigma}")
    preprocessAll(ROOT_PATH, expList, mean, sigma)
    preprocVis("/home/tzikos/Desktop/Data/Berts/AVNRT/AVNRT tachy/AVNRT tachy/75E6EFE9-4EC8-4B3A-9E89-5004E6A326F2_2AX1ZWAX3DRF_20220426_2.xlsx",
               mean[0], sigma[0])

if CALC_WEIGHTS_BERT:
     classWeights = classInstanceCalc(ROOT_PATH, WEIGHT_PATH, expList, EXPERIMENT)
classWeights = calcWeights(WEIGHT_PATH, EXPERIMENT)
print(classWeights)
model = ECGCNNClassifier(CLASSES_BERT)

if FINETUNING:
    # Datasets
    trainDataset = ECGDataset(f"{DIR_PATH}/train/", EXPERIMENT, CLASSES_BERT, classWeights)
    stratTrainSampler = WeightedRandomSampler(weights=trainDataset.samplerWeights(), num_samples=len(trainDataset) )
    trainLoader = DataLoader(trainDataset, batch_size=BATCH, 
                             #sampler=stratTrainSampler
                             shuffle=True
                             )
    valDataset = ECGDataset(f"{DIR_PATH}/val/", EXPERIMENT, CLASSES_BERT, classWeights)
    stratValSampler = WeightedRandomSampler(weights=valDataset.samplerWeights(), num_samples=len(valDataset) )
    valLoader = DataLoader(valDataset, batch_size=BATCH, 
                           #sampler=stratValSampler
                           shuffle=True
                           )
    filePath = train(model, trainLoader, valLoader, CLASSES_BERT, LEARNING_RATE,
          EPOCHS, classWeights, EARLY_PAT, REDLR_PAT, device, expList)
else:
    filePath = f"/home/tzikos/Desktop/weights/ECGCNNClassifier_07-03-24-12-13.pth"

if TEST:
    testDataset = ECGDataset(f"{DIR_PATH}/test/", EXPERIMENT, CLASSES_BERT, classWeights)
    testLoader = DataLoader(testDataset, batch_size=BATCH)
    test(model, testLoader, CLASSES_BERT, device, filePath, expList)