from models import ECGDataset, PTBDataset, train, test, ECGCNNClassifier, ECGSimpleClassifier
from torch.utils.data import DataLoader
from preprocessing import appendStratifiedFiles, downsamplerNoiseRemover, gaussianCalcBert, gaussianNormalizerMissingLeadCreatorBert
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from preprocPTBXL import readPTB, gaussianNormalizerPTB, dataSplitterPTB, createMissingLeadsPTB
import numpy as np
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# HYPERPARAMETERS
LEARNING_RATE_PTB = 0.0000003
LEARNING_RATE_BERT = 0.000001
BATCH = 64
L1 = 0.0001
L2 = 0.0001
A_BERT = 10
A_PTB = 1

# Stable
EPOCHS = 5
EARLY_PAT = 12
REDLR_PAT = 5

# GLOBALS
CLASSES_BERT = 5
CLASSES_PTB = 12
ROOT_DATA_PATH = '/home/tzikos/Desktop/Data'
EXPERIMENT = "tachy"
WEIGHT_PATH = "/home/tzikos/Desktop/weights/"
PREPROC_PTB = False
PREPROC_BERT = False
PRETRAIN = False
FINETUNE = True
VISUALIZE = False
USE_PRETRAINED = True

if PRETRAIN or FINETUNE:
    model = ECGCNNClassifier(CLASSES_BERT)

    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="KU AI Thesis",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate_PTB": LEARNING_RATE_PTB,
        "learning_rate_BERT": LEARNING_RATE_BERT,
        "batch_size": BATCH,
        "L1": L1,
        "L2": L2,
        "A_PTB": A_PTB,
        "A_BERT": A_BERT,
        "early_patience": EARLY_PAT,
        "reduce_lr_patience": REDLR_PAT,
        "architecture": model.__class__.__name__,
        "experiment": EXPERIMENT,
        "pretrain": PRETRAIN,
        "finetune": FINETUNE,
        "classes_PTB": CLASSES_PTB,
        "use_pretrained": USE_PRETRAINED}
    )



if PREPROC_PTB:
    metadataPath="/home/tzikos/Desktop/Data/PTBXL/ptbxl_database.csv", 
    dataPath = "/home/tzikos/Desktop/Data/PTBXL/records500/", 
    if CLASSES_PTB == 44:
        typeC = "AllDia"
        countDict = {"NDT":0, "NST":0, "DIG":0, "LNGQT":0, "NORM":0, "IMI":0, "ASMI":0,
                    "LVH":0, "LAFB":0, "ISC_":0, "IRBBB":0, "1AVB":0, "IVCD":0, "ISCAL":0,
                    "CRBBB":0, "CLBBB":0, "ILMI":0, "LAO/LAE":0, "AMI":0, "ALMI":0, "ISCIN":0,
                    "INJAS":0, "LMI":0, "ISCIL":0, "LPFB":0, "ISCAS":0, "INJAL":0, "ISCLA":0,
                    "RVH":0, "ANEUR":0, "RAO/RAE":0, "EL":0, "WPW":0, "ILBBB":0, "IPLMI":0,
                    "ISCAN":0, "IPMI":0, "SEHYP":0, "INJIN":0, "INJLA":0, "PMI":0, "3AVB":0,
                    "INJIL":0, "2AVB":0}
        classDict = {"NDT":"NDT", "NST":"NST", "DIG":"DIG", "LNGQT":"LNGQT", "NORM":"NORM", 
                    "IMI":"IMI", "ASMI":"ASMI", "LVH":"LVH", "LAFB":"LAFB", "ISC_":"ISC_", 
                    "IRBBB":"IRBB", "1AVB":"1AVB", "IVCD":"IVCD", "ISCAL":"ISCAL", "CRBBB":"CRBBB",
                    "CLBBB":"CLBBB", "ILMI":"ILMI", "LAO/LAE":"LAO/:LAE", "AMI":"AMI", "ALMI":"ALMI",
                    "ISCIN":"ISCIN", "INJAS":"INJAS", "LMI":"LMI", "ISCIL":"ISCIL", "LPFB":"LPFB", 
                    "ISCAS":"ISCAS", "INJAL":"INJAL", "ISCLA":"ISCLA", "RVH":"RVH", "ANEUR":"ANEUR", 
                    "RAO/RAE":"RAO/RAE", "EL":"EL", "WPW":"WPW", "ILBBB":"ILBB", "IPLMI":"IPLMI",
                    "ISCAN":"ISCAN", "IPMI":"IPMI", "SEHYP":"SEHYP", "INJIN":"INJIN", "INJLA":"INJLA",
                    "PMI":"PMI", "3AVB":"3AVB", "INJIL":"INJIN", "2AVB":"2AVB"}
    elif CLASSES_PTB == 5:
        typeC = "Diagnostic"
        # Initialize the dictionary to count the class instances
        countDict = {"NORM":0, "MI":0, "STTC":0, "CD":0, "HYP":0}
        # Initialize the dictionary to map the classes to superclasses
        classDict = {"NDT":"STTC", "NST":"STTC", "DIG":"STTC", "LNGQT":"STTC", "NORM":"NORM", 
                    "IMI":"MI", "ASMI":"MI", "LVH":"HYP", "LAFB":"CD", "ISC_":"STTC", 
                    "IRBBB":"CD", "1AVB":"CD", "IVCD":"CD", "ISCAL":"STTC", "CRBBB":"CD",
                    "CLBBB":"CD", "ILMI":"MI", "LAO/LAE":"HYP", "AMI":"MI", "ALMI":"MI",
                    "ISCIN":"STTC", "INJAS":"MI", "LMI":"MI", "ISCIL":"STTC", "LPFB":"CD", 
                    "ISCAS":"STTC", "INJAL":"MI", "ISCLA":"STTC", "RVH":"HYP", "ANEUR":"STTC", 
                    "RAO/RAE":"HYP", "EL":"STTC"    , "WPW":"CD", "ILBBB":"CD", "IPLMI":"MI",
                    "ISCAN":"STTC", "IPMI":"MI", "SEHYP":"HYP", "INJIN":"MI", "INJLA":"MI",
                    "PMI":"MI", "3AVB":"CD", "INJIL":"MI", "2AVB":"CD"}
    elif CLASSES_PTB == 12:
        typeC = "Rhythm"
        countDict = {"SR":0, "AFIB":0, "STACH":0, "SARRH":0, "SBRAD":0, "PACE":0,
                     "SVARR":0, "BIGU":0, "AFLT":0, "SVTAC":0, "PSVT":0, "TRIGU":0}
        classDict = {"SR":"SR", "AFIB":"AFIB", "STACH":"STACH", "SARRH":"SARRH", "SBRAD":"SBRAD",
                    "PACE":"PACE", "SVARR":"SVARR", "BIGU":"BIGU", "AFLT":"AFLT", "SVTAC":"SVTAC",
                    "PSVT":"PSVT", "TRIGU":"TRIGU"}
    savePath1 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} Downsampled/"
    savePath2 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} Denoised/"
    savePath3 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} Gaussian/"
    savePath4 = f"/home/tzikos/Desktop/Data/PTBXL {typeC} torch/"
    tempPath = readPTB(countDict, classDict, metadataPath, dataPath, savePath1, savePath2)
    trainList, valList = dataSplitterPTB(tempPath, list(classDict.values()))
    gaussianNormalizerPTB(trainList, savePath2, CLASSES_PTB, savePath3)
    createMissingLeadsPTB(trainList, np.load(f"{WEIGHT_PATH}PTBXLmean{CLASSES_PTB}.npy"), 
                          np.load(f"{WEIGHT_PATH}PTBXLsigma{CLASSES_PTB}.npy"), "train", typeC)
    createMissingLeadsPTB(valList, np.load(f"{WEIGHT_PATH}PTBXLmean{CLASSES_PTB}.npy"), 
                          np.load(f"{WEIGHT_PATH}PTBXLsigma{CLASSES_PTB}.npy"), "val", typeC)

expList = [f'normal {EXPERIMENT}', f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']
PTBList = ["NORM", "MI", "STTC", "CD", "HYP"]

if PREPROC_BERT:
    trainFiles, valFiles, testFiles = appendStratifiedFiles(f"{ROOT_DATA_PATH}/Berts/", expList)
    downsamplerNoiseRemover(trainFiles, f"{ROOT_DATA_PATH}/Berts downsapled/{EXPERIMENT}/train", 
                            f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/train")
    downsamplerNoiseRemover(valFiles, f"{ROOT_DATA_PATH}/Berts downsapled/{EXPERIMENT}/val", 
                            f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/val")    
    downsamplerNoiseRemover(testFiles, f"{ROOT_DATA_PATH}/Berts downsapled/{EXPERIMENT}/test", 
                            f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/test")
    print("End of downsampling and noise removal")
    mean, sigma = gaussianCalcBert(f"{ROOT_DATA_PATH}/Berts no noise/{EXPERIMENT}/train", EXPERIMENT)
    print(mean)
    print(mean.shape)
    print(sigma)
    print(sigma.shape)
    mean = np.load(f'{WEIGHT_PATH}meanBerts{EXPERIMENT}.npy')
    sigma = np.load(f'{WEIGHT_PATH}sigmaBerts{EXPERIMENT}.npy') 
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
if PRETRAIN:
    # Datasets
    classWeights = np.load(WEIGHT_PATH + "PTBweights.npy")
    classWeights = A_PTB * classWeights
    print(classWeights.shape)
    print(classWeights)
    trainDataset = PTBDataset(f"{ROOT_DATA_PATH}/PTBXL torch/train/", CLASSES_PTB, classWeights)
    trainLoader = DataLoader(trainDataset, batch_size=BATCH, shuffle=True, num_workers=8)
    valDataset = PTBDataset(f"{ROOT_DATA_PATH}/PTBXL torch/val/", CLASSES_PTB, classWeights)
    valLoader = DataLoader(valDataset, batch_size=BATCH, shuffle=True, num_workers=8)
    filePath = train(model, trainLoader, valLoader, CLASSES_PTB, LEARNING_RATE_PTB,
          EPOCHS, classWeights, EARLY_PAT, REDLR_PAT, device, PTBList,
          dataset="PTBXL", batchSize=BATCH, lr=LEARNING_RATE_PTB)


if FINETUNE:
    classWeights = np.load(f'{WEIGHT_PATH}Bert{EXPERIMENT}Weights.npy')
    classWeight = A_BERT * classWeights
    if USE_PRETRAINED:
        # Path to pretrained weights
        pretrainedWeightsPath = "/home/tzikos/Desktop/weights/ECGCNNClassifier_PTBXL_B64_L3e-07_13-03-24-23-28.pth"
        pretrainedWeights = torch.load(pretrainedWeightsPath)
        # Filter out the weights for the classification head
        # Adjusting the key names
        pretrainedWeights = {k: v for k, v in pretrainedWeights.items() if not k.startswith('fc')}
        model = ECGCNNClassifier(CLASSES_BERT)
        model.load_state_dict(pretrainedWeights, strict=False)
    else:
        model = ECGCNNClassifier(CLASSES_BERT)
    # Datasets
    trainDataset = ECGDataset(f"{ROOT_DATA_PATH}/Berts torch/{EXPERIMENT}/train/", EXPERIMENT, CLASSES_BERT, classWeights)
    trainLoader = DataLoader(trainDataset, batch_size=BATCH, 
                             shuffle=True, 
                             #sampler=WeightedRandomSampler(weights=trainDataset.samplerWeights(), num_samples=len(trainDataset)),
                             num_workers=8)
    valDataset = ECGDataset(f"{ROOT_DATA_PATH}/Berts torch/{EXPERIMENT}/val/", EXPERIMENT, CLASSES_BERT, classWeights)
    valLoader = DataLoader(valDataset, batch_size=BATCH, shuffle=True, num_workers=8)
    filePath = train(model, trainLoader, valLoader, CLASSES_BERT, LEARNING_RATE_BERT,
          EPOCHS, classWeights, EARLY_PAT, REDLR_PAT, device, expList,
          dataset=EXPERIMENT, batchSize=BATCH, lr=LEARNING_RATE_BERT, L1=L1, L2=L2)   
    testDataset = ECGDataset(f"{ROOT_DATA_PATH}/Berts torch/{EXPERIMENT}/test/", EXPERIMENT, CLASSES_BERT, classWeights)
    testLoader = DataLoader(testDataset, batch_size=BATCH)
    test(model, testLoader, CLASSES_BERT, device, filePath, expList)


run = wandb.finish()
