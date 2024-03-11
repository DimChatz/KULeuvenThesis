import wfdb
import pandas as pd
from preprocessing import noiseRemover, gaussianCalc
import numpy as np
import os
import ast

def getGaussianParamsPTB(dataPath="/home/tzikos/Desktop/Data/PTBXLProcessed/"):
    stacked = np.zeros((2500, 12))
    for root, dirs, files in os.walk(dataPath):
        for file in files:
            # Check that excel files are picked, 
            # they are not the general files for all the tests and 
            # that they are of the correct experiment type (pre, tachy)
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                stacked = np.vstack((stacked, tempArray))
    stacked = stacked[25:, :]
    mean = []
    sigma = []
    for i in range(12):
        mean.append(np.mean(stacked[:, i]))
        sigma.append(np.std(stacked[:, i]))
    return mean, sigma
    
def gaussianNormalizerPTB(dataPath="/home/tzikos/Desktop/Data/PTBXLProcessed/"):
    mean, sigma = getGaussianParamsPTB(dataPath)
    for root, dirs, files in os.walk(dataPath):
        for file in files:
            # Check that excel files are picked, 
            # they are not the general files for all the tests and 
            # that they are of the correct experiment type (pre, tachy)
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                for i in range(12):
                    tempArray[:, i] = (tempArray[:, i] - mean[i]) / sigma[i]
                np.save(os.path.join(root, file), tempArray)

def preprocPTBXL(metadataPath="/home/tzikos/Desktop/Data/PTBXL/ptbxl_database.csv", 
                 dataPath = "/home/tzikos/Desktop/Data/PTBXL/records500/", 
                 savePath = "/home/tzikos/Desktop/Data/PTBXLProcessed/"):
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
                if (key in classDict) and (scpDict[key]==100.0):
                    patientList.append(patID)
                    recordName = f'{dataPath}{str((ecgID // 1000) * 1000).zfill(5)}/{str(ecgID).zfill(5)}_hr'
                    record = wfdb.rdrecord(recordName, sampfrom=0, sampto=None, channels=None, physical=True)
                    # Accessing the signal itself
                    signals = record.p_signal
                    # Downsample to 250Hz
                    signals = signals[::2, :]
                    # Remove noise
                    signals = noiseRemover(signals)
                    np.save(f'{savePath}{classDict[key]}/{ecgID}.npy', signals)
                    countDict[classDict[key]] += 1
    for i in countDict.keys():
        print(f'{i} has {countDict[i]} instances')


