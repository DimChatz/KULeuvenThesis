import wfdb
import pandas as pd
from preprocessing import noiseRemover
import numpy as np
import os
import ast
from visualizer import VisNP

def getGaussianParamsPTB(dataPath="/home/tzikos/Desktop/Data/PTBXL npys/"):
    """Function to calculate the Gaussian parameters for the PTBXL dataset"""
    # Initialize the array to stack the data
    stacked = np.zeros((2500, 12))
    # Recursively search for excel files in the directory
    for root, dirs, files in os.walk(dataPath):
        for file in files:
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                # Stack the data
                stacked = np.vstack((stacked, tempArray))
    # Remove the first duplicate of zeros
    stacked = stacked[2500:, :]
    # Calculate the mean and sigma
    mean, sigma = [], []
    for i in range(12):
        mean.append(np.mean(stacked[:, i]))
        sigma.append(np.std(stacked[:, i]))
    return mean, sigma
    
def gaussianNormalizerPTB(dataPath="/home/tzikos/Desktop/Data/PTBXL npys/",
                          savePath="/home/tzikos/Desktop/Data/PTBXL normed/"):
    """Function to Gaussian normalize the PTBXL dataset"""
    mean, sigma = getGaussianParamsPTB(dataPath)
    mean = np.array(mean, dtype=np.float32)
    sigma = np.array(sigma, dtype=np.float32)
    # Save the mean and sigma
    # because it takes too long to calculate
    np.save("/home/tzikos/Desktop/weights/PTBXLmean.npy", mean)
    np.save("/home/tzikos/Desktop/weights/PTBXLsigma.npy", sigma)
    # print the mean and sigma 
    # for safety
    print(f'mean is {mean}')
    print(f'sigma is {sigma}')
    # Count for visualization
    count = 0
    # Recursively search for excel files in the directory
    for root, dirs, files in os.walk(dataPath):
        for file in files:
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                # If the count is less than 2
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
    """Function to create missing leads for the PTBXL dataset"""
    # Count for visualization
    count = 0
    for file in fileList:
        # Load file
        data = np.load(file)
        # Add dimension to add the data
        data = np.expand_dims(data, axis = -1)
        # Create missing leads
        endArray = data.copy()
        for i in range(data.shape[1]):
            tempArray = data.copy()
            tempArray[:, i, :] = 0.
            endArray = np.concatenate((endArray, tempArray), axis = 2)
        for i in range(endArray.shape[2]):
            if count == 0:
                VisNP(endArray[:,:,1], saveName="exampleAfterRemovingLead", comment="After removing Lead")
                count += 1
            np.save(f"/home/tzikos/Desktop/Data/PTBXL torch/{split}/{file.split("/")[-2]}-{file.split("/")[-1][:-4]}-{i}.npy", endArray[:,:,i])


def readPTB_5(metadataPath="/home/tzikos/Desktop/Data/PTBXL/ptbxl_database.csv", 
                 dataPath = "/home/tzikos/Desktop/Data/PTBXL/records500/", 
                 savePath = "/home/tzikos/Desktop/Data/PTBXL npys/"):
    """Function to read the PTBXL dataset and save the data as npy files"""
    # Initialize the dictionary to count the class instances
    countDict = {"NORM":0, "MI":0, "STTC":0, "CD":0, "HYP":0}
    # Initialize the dictionary to map the classes to superclasses
    classDict = {"NORM": "NORM", 
                "LMI":"MI", "PMI":"MI", "IMI":"MI", "AMI":"MI",
                "ISCI":"STTC", "NST_":"STTC", "ISCA":"STTC", "ISC_":"STTC", "STTC":"STTC", 
                "LAFB/LPFB":"CD", "IRBBB":"CD", "_AVB":"CD", "IVCD":"CD", "CRBBB":"CD", "CLBBB":"CD", "WPW":"CD", "ILBBB":"CD",  
                "LVH":"HYP", "RVH":"HYP", "LAO/LAE":"HYP", "RAO/RAE":"HYP", "SEHYP":"HYP"}
    metaData = pd.read_csv(metadataPath)
    # Initialize the list to save the patient IDs
    # and avoid duplicates
    patientList = []
    # Keep only the necessary columns
    metaData = metaData[["ecg_id", "patient_id", "scp_codes"]]
    for index, row in metaData.iterrows():
        patID = row["patient_id"]
        ecgID = int(row["ecg_id"])
        # Dict that contains the classes
        scpDict = ast.literal_eval(row["scp_codes"])
        # If the patient is not in the list
        if patID not in patientList:
            # For all classes in the ecg example
            for key in scpDict.keys():
                # If the class is in the classDict
                if (key in classDict):
                    # Add the patient to the list
                    # to avoid duplicates
                    patientList.append(patID)
                    # Load the record
                    recordName = f'{dataPath}{str((ecgID // 1000) * 1000).zfill(5)}/{str(ecgID).zfill(5)}_hr'
                    # Access the record
                    record = wfdb.rdrecord(recordName, sampfrom=0, sampto=None, channels=None, physical=True)
                    # Accessing the signal itself
                    signals = record.p_signal
                    # Downsample to 250Hz - all are 500Hz
                    signals = signals[::2, :]
                    # Remove noise and visualize
                    if ecgID == 1:
                        VisNP(signals, saveName="exampleBeforeNoiseRemoval", comment="Before noise removal")
                        signals = noiseRemover(signals)
                        VisNP(signals, saveName="exampleAfterNoiseRemoval", comment="After noise removal")
                    else:
                        signals = noiseRemover(signals)
                    np.save(f'{savePath}{classDict[key]}/{ecgID}.npy', signals)
                    # Add to the count of class instances
                    countDict[classDict[key]] += 1
    # Total number of samples                
    total = sum(countDict.values())  
    # Number of classes
    num_classes = len(countDict)  
    # Calculate the class weights
    classWeights = [total / (num_classes * countDict[i]) for i in countDict]
    classWeights = np.array(classWeights, dtype=np.float32)
    print(f"The class weights for PTB are of shape {classWeights.shape}")
    print(f"and are {classWeights}")
    np.save(f'/home/tzikos/Desktop/weights/PTBweights.npy', classWeights)


def dataSplitterPTB(directory="/home/tzikos/Desktop/Data/PTBXL normed/", segmentList=["NORM", "MI", "STTC", "CD", "HYP"]):  
    """Function to create balanced datasets"""
    # Initialize the lists to save the files 
    trainFiles, valFiles, = [], []
    # For all classes
    for i in range(len(segmentList)):
        # Initialize the list to save the npy files
        npyFiles = []
        for root, dirs, files in os.walk(f'{directory}/{segmentList[i]}'):
            for file in files:
                if file.endswith('.npy'):
                    npyFiles.append(os.path.join(root, file))
        # Split 90-10
        trainFiles += npyFiles[:int(np.round(0.9*len(npyFiles)))]
        valFiles += npyFiles[int(np.round(0.9*len(npyFiles))):]
    return trainFiles, valFiles

