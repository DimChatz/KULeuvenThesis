from models import ECGDataset, train
from torch.utils.data import DataLoader
from preprocessing import gaussianNormalizer, preprocessAll
from utility import calcWeights, classInstanceCalc

# HYPERPARAMETERS
LEARNING_RATE = 0.001
BATCH = 8
EPOCHS = 25
CLASSES = 4
EXPERIMENT = "pre"
DIR_PATH = f"/home/tzikos/Desktop/Data/Berts torch/{EXPERIMENT}"
PREPROC = False
CALC_WEIGHTS = True
ROOT_PATH = '/home/tzikos/Desktop/Data/Berts/'

expList = [f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']

if PREPROC:
    mean, sigma = gaussianNormalizer(ROOT_PATH, segment = f'{EXPERIMENT}')
    print(mean, sigma)
    preprocessAll(ROOT_PATH, expList)

if CALC_WEIGHTS:
     classWeights = classInstanceCalc(ROOT_PATH, DIR_PATH, expList, EXPERIMENT)
classWeights = calcWeights(DIR_PATH, EXPERIMENT)

# Datasets
trainDataset = ECGDataset(f"{DIR_PATH}/train/", EXPERIMENT, CLASSES)
trainLoader = DataLoader(trainDataset, batch_size=BATCH, shuffle=True)
valDataset = ECGDataset(f"{DIR_PATH}/val/", EXPERIMENT, CLASSES)
valLoader = DataLoader(valDataset, batch_size=BATCH, shuffle=True)
testDataset = ECGDataset(f"{DIR_PATH}/val/", EXPERIMENT, CLASSES)
testLoader = DataLoader(testDataset, batch_size=BATCH, shuffle=True)

train(trainLoader, valLoader, testLoader, LEARNING_RATE, EPOCHS, CLASSES, classWeights)