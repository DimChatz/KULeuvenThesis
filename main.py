from models import ECGDataset, train, ECGCNNClassifier, ECGSimpleClassifier
from torch.utils.data import DataLoader
from preprocessing import gaussianNormalizer, preprocessAll
from utility import calcWeights, classInstanceCalc

# HYPERPARAMETERS
LEARNING_RATE = 0.0001
BATCH = 64
EPOCHS = 1000
CLASSES = 4
EXPERIMENT = "pre"
DIR_PATH = f"/home/tzikos/Desktop/Data/Berts torch/{EXPERIMENT}"
PREPROC = False
TRAIN = True
CALC_WEIGHTS = True
ROOT_PATH = '/home/tzikos/Desktop/Data/Berts/'
EARLY_PAT = 12
REDLR_PAT = 5

expList = [f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']

if PREPROC:
    mean, sigma = gaussianNormalizer(ROOT_PATH, segment = f'{EXPERIMENT}')
    print(mean, sigma)
    preprocessAll(ROOT_PATH, expList, mean, sigma)

if CALC_WEIGHTS:
     classWeights = classInstanceCalc(ROOT_PATH, DIR_PATH, expList, EXPERIMENT)
classWeights = calcWeights(DIR_PATH, EXPERIMENT)
print(classWeights)

if TRAIN:
    # Datasets
    trainDataset = ECGDataset(f"{DIR_PATH}/train/", EXPERIMENT, CLASSES)
    trainLoader = DataLoader(trainDataset, batch_size=BATCH, shuffle=True)
    valDataset = ECGDataset(f"{DIR_PATH}/val/", EXPERIMENT, CLASSES)
    valLoader = DataLoader(valDataset, batch_size=BATCH, shuffle=True)
    testDataset = ECGDataset(f"{DIR_PATH}/val/", EXPERIMENT, CLASSES)
    testLoader = DataLoader(testDataset, batch_size=BATCH, shuffle=True)

    train(ECGCNNClassifier(CLASSES), trainLoader, valLoader, testLoader, CLASSES, LEARNING_RATE, EPOCHS, classWeights, EARLY_PAT, REDLR_PAT)