from models import ECGDataset, train, test, ECGCNNClassifier, ECGSimpleClassifier
from torch.utils.data import DataLoader
from preprocessing import gaussianNormalizer, preprocessAll
from utility import calcWeights, classInstanceCalc
from visualizer import preprocVis
from torch.utils.data.sampler import WeightedRandomSampler
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# HYPERPARAMETERS
LEARNING_RATE = 0.000003
BATCH = 256
EPOCHS = 100
EARLY_PAT = 12
REDLR_PAT = 5
# GLOBALS
CLASSES = 4
EXPERIMENT = "pre"
DIR_PATH = f"/home/tzikos/Desktop/Data/Berts torch/{EXPERIMENT}"
WEIGHT_PATH = "/home/tzikos/Desktop/weights/"
PREPROC = False
CALC_WEIGHTS = False
TRAIN = True
TEST = True
ROOT_PATH = '/home/tzikos/Desktop/Data/Berts/'


expList = [f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']

if PREPROC:
    mean, sigma = gaussianNormalizer(ROOT_PATH, segment = f'{EXPERIMENT}')
    print(f"mean is {mean}, and sigma is {sigma}")
    preprocessAll(ROOT_PATH, expList, mean, sigma)
    preprocVis("/home/tzikos/Desktop/Data/Berts/AVNRT/AVNRT tachy/AVNRT tachy/75E6EFE9-4EC8-4B3A-9E89-5004E6A326F2_2AX1ZWAX3DRF_20220426_2.xlsx",
               mean, sigma)

if CALC_WEIGHTS:
     classWeights = classInstanceCalc(ROOT_PATH, WEIGHT_PATH, expList, EXPERIMENT)
classWeights = calcWeights(WEIGHT_PATH, EXPERIMENT)
print(classWeights)
model = ECGCNNClassifier(CLASSES)

if TRAIN:
    # Datasets
    trainDataset = ECGDataset(f"{DIR_PATH}/train/", EXPERIMENT, CLASSES, classWeights)
    stratTrainSampler = WeightedRandomSampler(weights=trainDataset.samplerWeights(), num_samples=len(trainDataset) )
    trainLoader = DataLoader(trainDataset, batch_size=BATCH, sampler=stratTrainSampler)
    valDataset = ECGDataset(f"{DIR_PATH}/val/", EXPERIMENT, CLASSES, classWeights)
    stratValSampler = WeightedRandomSampler(weights=valDataset.samplerWeights(), num_samples=len(valDataset) )
    valLoader = DataLoader(valDataset, batch_size=BATCH, sampler=stratValSampler)
    filePath = train(model, trainLoader, valLoader, CLASSES, LEARNING_RATE,
          EPOCHS, classWeights, EARLY_PAT, REDLR_PAT, device, expList)
else:
    filePath = f"/home/tzikos/Desktop/weights/ECGCNNClassifier_07-03-24-12-13.pth"

if TEST:
    testDataset = ECGDataset(f"{DIR_PATH}/test/", EXPERIMENT, CLASSES, classWeights)
    stratTestSampler = WeightedRandomSampler(weights=testDataset.samplerWeights(), num_samples=len(testDataset) )
    testLoader = DataLoader(testDataset, batch_size=BATCH, sampler=stratTestSampler)
    test(model, testLoader, CLASSES, device, filePath, expList)