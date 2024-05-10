import wfdb
import pandas as pd
from preprocessing import noiseRemover, resampler
import numpy as np
import os
import ast
from tqdm import tqdm


def getGaussianParamsPTB(fileList):
    """Function to calculate the Gaussian parameters for the PTBXL dataset"""
    meanArray = np.zeros(12, dtype=np.float32)
    sigmaArray = np.zeros(12, dtype=np.float32)
    # Recursively search for excel files in the directory
    for i in range(12):
        # Initialize the array to stack the data
        stacked = np.zeros((5000,1), dtype=np.float32)
        for file in tqdm(fileList):
            tempArray = np.expand_dims(np.load(file)[:,i], axis=-1)
            # Stack the data
            stacked = np.concatenate((stacked, tempArray), axis=1)
            #print(stacked.shape)
        # Remove the first duplicate of zeros
        stacked = stacked[:, 1:]
        # Calculate the mean and sigma
        mean = np.mean(stacked)
        sigma = np.std(stacked)
        print(f"Mean is {mean}")
        print(f"Sigma is {sigma}")
        meanArray[i] = mean
        sigmaArray[i] = sigma
    np.save(f'/home/tzikos/Desktop/weights/meanPTB.npy', meanArray)
    np.save(f'/home/tzikos/Desktop/weights/sigmaPTB.npy', sigmaArray)    

def gaussianNormalizerPTB(trainList, valList, savePath1, savePath2):
    """Function to Gaussian normalize the PTBXL dataset"""
    mean = np.load(f'/home/tzikos/Desktop/weights/meanPTB.npy')
    sigma = np.load(f'/home/tzikos/Desktop/weights/sigmaPTB.npy')
    # Recursively search for excel files in the directory
    for file in tqdm(trainList):            
        tempArray = np.load(file)
        tempArray = (tempArray - mean) / sigma
        np.save(os.path.join(savePath1, f"train/{file.split("/")[-1]}"), tempArray.astype(np.float32))
    for file in tqdm(valList):
        tempArray = np.load(file)
        tempArray = (tempArray - mean) / sigma
        np.save(os.path.join(savePath2, f"val/{file.split("/")[-1]}"), tempArray.astype(np.float32))      


def createMissingLeadsPTB(dataPath, savePath):
    """Function to create missing leads for the PTBXL dataset"""
    # Count for visualization
    trainFileList = os.listdir(f"{dataPath}train")
    trainFileList = [os.path.join(f"{dataPath}train", file) for file in trainFileList]
    for file in tqdm(trainFileList):
        openArray = np.load(file)
        np.save(os.path.join(savePath, f"train/{file.split('/')[-1]}"), openArray.astype(np.float32))
        for i in range(12):
            tempArray = openArray.copy()
            tempArray[:, i] = 0
            np.save(os.path.join(savePath, f"train/missingLead{i+1}-{file.split('/')[-1]}"), tempArray.astype(np.float32))


def readPTB(countDict, classDict, metadataPath, dataPath, savePath):
    """Function to read the PTBXL dataset and save the data as npy files"""
    metaData = pd.read_csv(metadataPath)
    # Initialize the list to save the patient IDs
    # and avoid duplicates
    patientList = []
    count=0
    # Keep only the necessary columns
    metaData = metaData[["ecg_id", "patient_id", "scp_codes", "validated_by_human"]]
    for index, row in tqdm(metaData.iterrows()):
        patID = int(row["patient_id"])
        ecgID = int(row["ecg_id"])
        # Dict that contains the classes
        scpDict = ast.literal_eval(row["scp_codes"])
        # For all classes in the ecg example
        if patID not in patientList:
            # If the patient is not in the list
            # Add the patient to the list
            # to avoid duplicates
            patientList.append(patID)
            for key in scpDict.keys():
                # If the class is in the classDict
                if (key in classDict) and (scpDict[key] >= 80.):
                    # Add the patient to the list
                    # to avoid duplicates
                    patientList.append(patID)
                    # Load the record
                    recordName = f'{dataPath}{str((ecgID // 1000) * 1000).zfill(5)}/{str(ecgID).zfill(5)}_hr'
                    # Access the record
                    record = wfdb.rdrecord(recordName, sampfrom=0, sampto=None, channels=None, physical=True)
                    # Accessing the signal itself
                    signals = record.p_signal
                    # Remove noise
                    signals = noiseRemover(signals, highF=100, samplingF=500)
                    # Save Denoised
                    np.save(f'{savePath}{classDict[key]}/{classDict[key]}-{ecgID}.npy', signals.astype(np.float32))
                    # Add to the count of class instances
    for key in countDict.keys():
        countDict[key] = len(os.listdir(f"{savePath}{key}"))
        print(f"Class {key} has {countDict[key]} samples")
    countDict["NORM"] = np.max(list(countDict.values())[1:])
    print(f"Class NORM has {countDict['NORM']} samples")
    # Number of classes
    num_classes = len(countDict)
    # Total number of samples                
    total = sum(countDict.values())
    # Calculate the class weights
    classWeights = [total / (num_classes * countDict[i]) for i in countDict]
    classWeights = np.array(classWeights, dtype=np.float32)
    print(f"The class weights for PTB are of shape {classWeights.shape}")
    print(f"and are {classWeights}")
    np.save(f'/home/tzikos/Desktop/weights/PTBweights{num_classes}.npy', classWeights.astype(np.float32))
    print("For best data splitting, the number of NORM samples is", countDict["NORM"])
    return countDict["NORM"]

def dataSplitterPTB(directory, segmentList, countNorm):  
    """Function to create balanced datasets"""
    # Initialize the lists to save the files
    trainFiles, valFiles, = [], []
    # For all classes
    for i in range(len(segmentList)):
        # Initialize the list to save the npy files
        npyFiles = os.listdir(f"{directory}/{segmentList[i]}")
        if "NORM" in segmentList[i]:
            npyFiles = npyFiles[:countNorm]
        origFiles = [os.path.join(f"{directory}/{segmentList[i]}", file) for file in npyFiles]
        trainAdd = origFiles[:int(np.round(0.8*len(origFiles)))]
        valAdd = origFiles[int(np.round(0.8*len(origFiles))):]
        # Split 90-10
        trainFiles += trainAdd
        valFiles += valAdd
    return trainFiles, valFiles