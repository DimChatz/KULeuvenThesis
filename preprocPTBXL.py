import wfdb
import pandas as pd
from preprocessing import noiseRemover
import numpy as np
import os
import ast
from visualizer import VisNP

def getGaussianParamsPTB(dataPath="/home/tzikos/Desktop/Data/PTBXL npys/"):
    stacked = np.zeros((2500, 12))
    count = 0
    for root, dirs, files in os.walk(dataPath):
        for file in files:
            # Check that excel files are picked, 
            # they are not the general files for all the tests and 
            # that they are of the correct experiment type (pre, tachy)
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                stacked = np.vstack((stacked, tempArray))
                count += 1
                #print(count)
    stacked = stacked[2500:, :]
    mean = []
    sigma = []
    for i in range(12):
        mean.append(np.mean(stacked[:, i]))
        sigma.append(np.std(stacked[:, i]))
    return mean, sigma
    
def gaussianNormalizerPTB(dataPath="/home/tzikos/Desktop/Data/PTBXL npys/",
                          savePath="/home/tzikos/Desktop/Data/PTBXL normed/"):
    mean, sigma = getGaussianParamsPTB(dataPath)
    mean = np.array(mean, dtype=np.float32)
    sigma = np.array(sigma, dtype=np.float32)
    np.save("/home/tzikos/Desktop/weights/PTBXLmean.npy", mean)
    np.save("/home/tzikos/Desktop/weights/PTBXLsigma.npy", sigma)
    print(f'mean is {mean}')
    print(f'sigma is {sigma}')
    count = 0
    for root, dirs, files in os.walk(dataPath):
        for file in files:
            # Check that excel files are picked, 
            # they are not the general files for all the tests and 
            # that they are of the correct experiment type (pre, tachy)
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                if count <2:
                    VisNP(tempArray, saveName="exampleBeforeNorm", comment="Before normalization")
                    count += 1
                for i in range(12):
                    tempArray[:, i] = (tempArray[:, i] - mean[i]) / sigma[i]
                if count <2:
                    VisNP(tempArray, saveName="exampleAfterNorm", comment="After normalization")
                    count += 1
                np.save(os.path.join(savePath, f"{root.split("/")[-1]}/{file}"), tempArray)


def createMissingLeadsPTB(fileList, split):
    count = 0
    for file in fileList:
        data = np.load(file)
        data = np.expand_dims(data, axis = -1)
        endArray = data.copy()
        for i in range(data.shape[1]):
            tempArray = data.copy()
            tempArray[:, i, :] = 0.
            endArray = np.concatenate((endArray, tempArray), axis = 2)
            #print(f"/home/tzikos/Desktop/Data/PTBXL torch/{split}/{file.split("/")[-2]}-{file.split("/")[-1][:-4]}.npy")
        for i in range(endArray.shape[2]):
            if count == 0:
                VisNP(endArray[:,:,1], saveName="exampleAfterRemovingLead", comment="After removing Lead")
                count += 1
            np.save(f"/home/tzikos/Desktop/Data/PTBXL torch/{split}/{file.split("/")[-2]}-{file.split("/")[-1][:-4]}-{i}.npy", endArray[:,:,i])


def readPTB_5(metadataPath="/home/tzikos/Desktop/Data/PTBXL/ptbxl_database.csv", 
                 dataPath = "/home/tzikos/Desktop/Data/PTBXL/records500/", 
                 savePath = "/home/tzikos/Desktop/Data/PTBXL npys/"):
    countDict = {"NORM":0, "MI":0, "STTC":0, "CD":0, "HYP":0}
    classDict = {"NORM": "NORM", 
                "LMI":"MI", "PMI":"MI", "IMI":"MI", "AMI":"MI",
                "ISCI":"STTC", "NST_":"STTC", "ISCA":"STTC", "ISC_":"STTC", "STTC":"STTC", 
                "LAFB/LPFB":"CD", "IRBBB":"CD", "_AVB":"CD", "IVCD":"CD", "CRBBB":"CD", "CLBBB":"CD", "WPW":"CD", "ILBBB":"CD",  
                "LVH":"HYP", "RVH":"HYP", "LAO/LAE":"HYP", "RAO/RAE":"HYP", "SEHYP":"HYP"}
    metaData = pd.read_csv(metadataPath)
    patientList = []
    metaData = metaData[["ecg_id", "patient_id", "scp_codes"]]
    for index, row in metaData.iterrows():
        patID = row["patient_id"]
        ecgID = int(row["ecg_id"])
        scpDict = ast.literal_eval(row["scp_codes"])
        if patID not in patientList:
            for key in scpDict.keys():
                if (key in classDict):
                    patientList.append(patID)
                    recordName = f'{dataPath}{str((ecgID // 1000) * 1000).zfill(5)}/{str(ecgID).zfill(5)}_hr'
                    record = wfdb.rdrecord(recordName, sampfrom=0, sampto=None, channels=None, physical=True)
                    # Accessing the signal itself
                    signals = record.p_signal
                    # Downsample to 250Hz
                    signals = signals[::2, :]
                    # Remove noise
                    if ecgID == 1:
                        VisNP(signals, saveName="exampleBeforeNoiseRemoval", comment="Before noise removal")
                        signals = noiseRemover(signals)
                        VisNP(signals, saveName="exampleAfterNoiseRemoval", comment="After noise removal")
                    else:
                        signals = noiseRemover(signals)
                    np.save(f'{savePath}{classDict[key]}/{ecgID}.npy', signals)
                    countDict[classDict[key]] += 1
    # Assume countDict is correctly populated with class instance counts.
    total = sum(countDict.values())  # Total number of samples
    num_classes = len(countDict)  # Number of classes
    classWeights = [total / (num_classes * countDict[i]) for i in countDict]
    classWeights = np.array(classWeights, dtype=np.float32)
    print(f"The class weights for PTB are {classWeights.shape}")
    np.save(f'/home/tzikos/Desktop/weights/PTBweights.npy', classWeights)

def readPTB_23(metadataPath="/home/tzikos/Desktop/Data/PTBXL/ptbxl_database.csv", 
                 dataPath = "/home/tzikos/Desktop/Data/PTBXL/records500/", 
                 savePath = "/home/tzikos/Desktop/Data/PTBXL npys/"):
    countDict = {"NORM":0, "LMI":0, "PMI":0, "IMI":0, "AMI":0, "ISCI":0, "NST_":0, "ISCA":0, "ISC_":0, "STTC":0,
                 "LAFB/LPFB":0, "IRBBB":0, "_AVB":0, "IVCD":0, "CRBBB":0, "CLBBB":0, "WPW":0, "ILBBB":0,  
                 "LVH":0, "RVH":0, "LAO/LAE":0, "RAO/RAE":0, "SEHYP":0}
    classList = {"NORM", "LMI", "PMI", "IMI", "AMI", "ISCI", "NST_", "ISCA", "ISC_", "STTC", 
                "LAFB/LPFB", "IRBBB", "_AVB", "IVCD", "CRBBB", "CLBBB", "WPW", "ILBBB",  
                "LVH", "RVH", "LAO/LAE", "RAO/RAE", "SEHYP"}
    metaData = pd.read_csv(metadataPath)
    patientList = []
    metaData = metaData[["ecg_id", "patient_id", "scp_codes"]]
    for index, row in metaData.iterrows():
        patID = row["patient_id"]
        ecgID = int(row["ecg_id"])
        scpDict = ast.literal_eval(row["scp_codes"])
        if patID not in patientList:
            for key in scpDict.keys():
                if (key in classList):
                    patientList.append(patID)
                    recordName = f'{dataPath}{str((ecgID // 1000) * 1000).zfill(5)}/{str(ecgID).zfill(5)}_hr'
                    record = wfdb.rdrecord(recordName, sampfrom=0, sampto=None, channels=None, physical=True)
                    # Accessing the signal itself
                    signals = record.p_signal
                    # Downsample to 250Hz
                    signals = signals[::2, :]
                    # Remove noise
                    if ecgID == 1:
                        VisNP(signals)
                        signals = noiseRemover(signals)
                        VisNP(signals)
                    else:
                        signals = noiseRemover(signals)
                    np.save(f'{savePath}{key}/{ecgID}.npy', signals)
                    countDict[key] += 1
    # Assume countDict is correctly populated with class instance counts.
    total = sum(countDict.values())  # Total number of samples
    num_classes = len(countDict)  # Number of classes

    classWeights = [total / (num_classes * countDict[i]) for i in countDict]
    classWeights = np.array(classWeights, dtype=np.float32)
    print(classWeights.shape)
    np.save(f'/home/tzikos/Desktop/weights/PTBweights.npy', classWeights)
    return classWeights


def dataSplitterPTB(directory="/home/tzikos/Desktop/Data/PTBXL normed/", segmentList=["NORM", "MI", "STTC", "CD", "HYP"]):   
    trainFiles, valFiles, = [], []
    # For all classes
    for i in range(len(segmentList)):
        npyFiles = []
        for root, dirs, files in os.walk(f'{directory}/{segmentList[i]}'):
            for file in files:
                # Check that excel files are picked, 
                # they are not the general files for all the tests and 
                # that they are of the correct experiment type (pre, tachy) 
                if file.endswith('.npy'):
                    npyFiles.append(os.path.join(root, file))
        # Split 80-10-10
        trainFiles += npyFiles[:int(np.round(0.9*len(npyFiles)))]
        valFiles += npyFiles[int(np.round(0.9*len(npyFiles))):]
    return trainFiles, valFiles

