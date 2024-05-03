import wfdb
import pandas as pd
from preprocessing import noiseRemover
import numpy as np
import os
import ast
from tqdm import tqdm

def getGaussianParamsPTB(fileList):
    """Function to calculate the Gaussian parameters for the PTBXL dataset"""
    # Initialize the array to stack the data
    stacked = np.zeros((5000, 12,))
    # Recursively search for excel files in the directory
    newFileList = [file for file in fileList if "orig" in file and "NORM" in file]
    print(len(newFileList))
    for file in tqdm(newFileList):
        tempArray = np.load(file)
        # Stack the data
        stacked = np.vstack((stacked, tempArray))
    # Remove the first duplicate of zeros
    stacked = stacked[5000:, :]
    # Calculate the mean and sigma
    mean = np.mean(stacked, axis=0)
    sigma = np.std(stacked, axis=0)
    np.save(f'/home/tzikos/Desktop/weights/meanPTB.npy', mean)
    np.save(f'/home/tzikos/Desktop/weights/sigmaPTB.npy', sigma)
    return mean, sigma
    

def gaussianNormalizerPTB(trainList, valList, segment, savePath):
    """Function to Gaussian normalize the PTBXL dataset"""
    mean = np.load(f'/home/tzikos/Desktop/weights/meanPTB.npy')
    sigma = np.load(f'/home/tzikos/Desktop/weights/sigmaPTB.npy')
    # Recursively search for excel files in the directory
    tempTrain = [file for file in trainList if segment in file]
    tempVal = [file for file in valList if segment in file]
    for file in tqdm(tempTrain):            
        tempArray = np.load(file)
        tempArray = (tempArray - mean) / sigma
        np.save(os.path.join(savePath, f"train/{file.split("/")[-1]}"), tempArray.astype(np.float32))
    for file in tqdm(tempVal):
        tempArray = np.load(file)
        tempArray = (tempArray - mean) / sigma
        np.save(os.path.join(savePath, f"val/{file.split("/")[-1]}"), tempArray.astype(np.float32))      


def createMissingLeadsPTB(dataPath, savePath):
    """Function to create missing leads for the PTBXL dataset"""
    # Count for visualization
    trainFileList = os.listdir(f"{dataPath}train")
    trainFileList = [os.path.join(f"{dataPath}train", file) for file in trainFileList]
    valFileList = os.listdir(f"{dataPath}val")
    valFileList = [os.path.join(f"{dataPath}val", file) for file in valFileList if "orig" in file]
    for file in tqdm(trainFileList):
        openArray = np.load(file)
        np.save(os.path.join(savePath, f"train/{file.split('/')[-1]}"), openArray.astype(np.float32))
        for i in range(12):
            tempArray = openArray.copy()
            tempArray[:, i] = 0
            np.save(os.path.join(savePath, f"train/missingLead{i+1}-{file.split('/')[-1]}"), tempArray.astype(np.float32))
    for file in tqdm(valFileList):
        openArray = np.load(file)
        np.save(os.path.join(savePath, f"val/{file.split('/')[-1]}"), openArray.astype(np.float32))

def readPTB(countDict, classDict, metadataPath, dataPath, savePath):
    """Function to read the PTBXL dataset and save the data as npy files"""
    metaData = pd.read_csv(metadataPath)
    # Initialize the list to save the patient IDs
    # and avoid duplicates
    patientList = []
    # Keep only the necessary columns
    metaData = metaData[["ecg_id", "patient_id", "scp_codes", "validated_by_human"]]
    for index, row in metaData.iterrows():
        patID = row["patient_id"]
        ecgID = int(row["ecg_id"])
        valid = row["validated_by_human"]
        # Dict that contains the classes
        scpDict = ast.literal_eval(row["scp_codes"])
        # If the patient is not in the list
        if patID not in patientList:
            # For all classes in the ecg example
            for key in scpDict.keys():
                # If the class is in the classDict
                if (key in classDict) and (scpDict[key] >= 80.0 or valid):
                    # Add the patient to the list
                    # to avoid duplicates
                    patientList.append(patID)
                    # Load the record
                    recordName = f'{dataPath}{str((ecgID // 1000) * 1000).zfill(5)}/{str(ecgID).zfill(5)}_hr'
                    # Access the record
                    record = wfdb.rdrecord(recordName, sampfrom=0, sampto=None, channels=None, physical=True)
                    # Accessing the signal itself
                    signals = record.p_signal
                    # Save original
                    np.save(f'{savePath}{classDict[key]}/orig-{classDict[key]}-{ecgID}.npy', signals)
                    # Remove noise and visualize
                    signals = noiseRemover(signals)
                    # Save Denoised
                    np.save(f'{savePath}{classDict[key]}/denoised-{classDict[key]}-{ecgID}.npy', signals)
                    # Add to the count of class instances
                    countDict[classDict[key]] += 1
    # Total number of samples                
    total = sum(countDict.values())
    for i in countDict.keys():
        print(f"Class {i} has {countDict[i]} samples")  
    # Number of classes
    num_classes = len(countDict)  
    # Calculate the class weights
    classWeights = [total / (num_classes * countDict[i]) for i in countDict]
    classWeights = np.array(classWeights, dtype=np.float32)
    print(f"The class weights for PTB are of shape {classWeights.shape}")
    print(f"and are {classWeights}")
    np.save(f'/home/tzikos/Desktop/weights/PTBweights{num_classes}.npy', classWeights)
    

def dataSplitterPTB(directory, segmentList):  
    """Function to create balanced datasets"""
    # Initialize the lists to save the files
    trainFiles, valFiles, = [], []
    # For all classes
    for i in range(len(segmentList)):
        # Initialize the list to save the npy files
        npyFiles = os.listdir(f"{directory}/{segmentList[i]}")
        origFiles = [os.path.join(f"{directory}/{segmentList[i]}", file) for file in npyFiles if "orig" in file]
        trainAdd = origFiles[:int(np.round(0.85*len(origFiles)))]
        valAdd = origFiles[int(np.round(0.85*len(origFiles))):]
        trainDenoisedAdd = [file.replace("orig", "denoised") for file in trainAdd]
        # Split 90-10
        trainFiles += trainAdd
        trainFiles += trainDenoisedAdd
        valFiles += valAdd
    return trainFiles, valFiles