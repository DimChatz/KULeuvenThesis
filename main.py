from models import ECGDataset, train, ECGCNNClassifier, ECGSimpleClassifier
from torch.utils.data import DataLoader
from preprocessing import gaussianNormalizer, preprocessAll
from utility import calcWeights, classInstanceCalc
from visualizer import preprocVis
from torch.utils.data.sampler import WeightedRandomSampler
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# HYPERPARAMETERS
LEARNING_RATE = 0.000001
BATCH = 256
EPOCHS = 1000
CLASSES = 4
EXPERIMENT = "tachy"
DIR_PATH = f"/home/tzikos/Desktop/Data/Berts torch/{EXPERIMENT}"
PREPROC = False
TRAIN = True
CALC_WEIGHTS = False
ROOT_PATH = '/home/tzikos/Desktop/Data/Berts/'
EARLY_PAT = 12
REDLR_PAT = 5

expList = [f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']

if PREPROC:
    mean, sigma = gaussianNormalizer(ROOT_PATH, segment = f'{EXPERIMENT}')
    print(mean, sigma)
    preprocessAll(ROOT_PATH, expList, mean, sigma)
    preprocVis("/home/tzikos/Desktop/Data/Berts/AVNRT/AVNRT tachy/AVNRT tachy/75E6EFE9-4EC8-4B3A-9E89-5004E6A326F2_2AX1ZWAX3DRF_20220426_2.xlsx",
               mean, sigma)

if CALC_WEIGHTS:
     classWeights = classInstanceCalc(ROOT_PATH, DIR_PATH, expList, EXPERIMENT)
classWeights = calcWeights(DIR_PATH, EXPERIMENT)
print(classWeights)

if TRAIN:
    # Datasets
    trainDataset = ECGDataset(f"{DIR_PATH}/train/", EXPERIMENT, CLASSES, classWeights)
    stratTrainSampler = WeightedRandomSampler(weights=trainDataset.samplerWeights(), num_samples=len(trainDataset) )
    trainLoader = DataLoader(trainDataset, batch_size=BATCH, sampler=stratTrainSampler)
    valDataset = ECGDataset(f"{DIR_PATH}/val/", EXPERIMENT, CLASSES, classWeights)
    stratValSampler = WeightedRandomSampler(weights=valDataset.samplerWeights(), num_samples=len(valDataset) )
    valLoader = DataLoader(valDataset, batch_size=BATCH, sampler=stratValSampler)
    testDataset = ECGDataset(f"{DIR_PATH}/test/", EXPERIMENT, CLASSES, classWeights)
    stratTestSampler = WeightedRandomSampler(weights=testDataset.samplerWeights(), num_samples=len(testDataset) )
    testLoader = DataLoader(testDataset, batch_size=BATCH, sampler=stratTestSampler)

    train(ECGCNNClassifier(CLASSES), trainLoader, valLoader, testLoader, CLASSES, LEARNING_RATE, EPOCHS, classWeights, EARLY_PAT, REDLR_PAT, device)