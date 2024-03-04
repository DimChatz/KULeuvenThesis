import os
import pandas as pd
import numpy as np
import torch
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")

# Read the data from the xlsx file
def downsampler(data):
    # Determine the current sampling frequency
    current_frequency = round(1 / data[data.columns[0]].iloc[1] - data[data.columns[0]].iloc[0])
    # Downsample if necessary
    if current_frequency == 500:
        # Take every other row to achieve 250Hz from 500Hz
        data = data.iloc[::2].reset_index(drop=True)
    elif current_frequency == 1000:
        data = data.iloc[::4].reset_index(drop=True)
    return data

# Function to recursively search for Excel files in a directory
def appendExcelFiles(directory, segment):
    excelFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx') and ("overzicht" not in file) and (segment in os.path.join(root, file)):
                excelFiles.append(os.path.join(root, file))
    return excelFiles

def appendStratifiedFiles(directory, segmentList):
    excelFiles = []
    trainFiles, valFiles, testFiles = [], [], []
    for i in range(len(segmentList)):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.xlsx') and ("overzicht" not in file) and (segmentList[i] in os.path.join(root, file)):
                    excelFiles.append(os.path.join(root, file))
        trainFiles += excelFiles[:int(np.round(0.8*len(excelFiles)))]
        valFiles += excelFiles[int(np.round(0.8*len(excelFiles))):int(np.round(0.9*len(excelFiles)))]
        testFiles += excelFiles[int(np.round(0.9*len(excelFiles))):]
    return trainFiles, valFiles, testFiles

# Min Max finder
def globalNormalizer(directory):
    endMin = 1000
    endMax = -1000
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx') and ("overzicht" not in file):
                xlData = pd.read_excel(f"{root}/{file}")
                xlData = downsampler(xlData)
                xlData = xlData.drop(xlData.columns[0], axis=1)
                tempMax = xlData.values.max()
                tempMin = xlData.values.min()
                endMax = max(tempMax, endMax)
                endMin = min(tempMin, endMin)
    return max(abs(endMin), endMax)

def gaussianCalc(gaussianArray):
    gaussianArray = gaussianArray[:, :, 1:]
    mean = np.mean(gaussianArray)
    sigma = np.std(gaussianArray)
    return mean, sigma

def gaussianNormalizer(folderPath, segment):
    gaussianArray = np.zeros((2500,12,1), np.float32)
    excelFiles = appendExcelFiles(folderPath, segment)
    for excelFile in excelFiles:
        tempData = pd.read_excel(excelFile)
        tempData = downsampler(tempData)
        tempData = tempData.drop(tempData.columns[0], axis=1)
        tempData = tempData.to_numpy()
        tempData = noiseRemover(tempData)
        tempExpanded = np.expand_dims(tempData, axis = -1)
        gaussianArray = np.concatenate((gaussianArray, tempExpanded), axis=2)
    return gaussianCalc(gaussianArray)
    
def createMissingLeads(startArray):
    endArray = startArray
    for i in range(startArray.shape[1]):
        tempArray = startArray.copy()
        tempArray[:, i, :] = 0.
        endArray = np.concatenate((endArray, tempArray), axis = 2)
    return endArray

def bandpass(lowF = 0.5, highF = 45, samplingF = 250, order = 5):
    nyqF = 0.5 * samplingF
    lowCut = lowF / nyqF
    highCut = highF / nyqF
    b, a = butter(order, [lowCut, highCut], btype='bandpass')
    return b, a

def noiseRemover(data, lowF = 0.5, highF = 45, samplingF = 250, order = 5):
    b, a = bandpass(lowF, highF, samplingF, order)
    y = filtfilt(b, a, data, axis = 0)
    return y

def preprocessAll(folderpath, segmentList):
    trainData, valData, testData = appendStratifiedFiles(folderpath, segmentList)
    print(len(trainData), len(valData), len(testData))
    segment = segmentList[0].split(" ")[-1]
    preprocPipeline(trainData, "train", segment)
    preprocPipeline(valData, "val", segment)
    preprocPipeline(testData, "test", segment)

def preprocPipeline(files, usage, segment, mean, sigma):
    for file in files:
        data = pd.read_excel(file)
        tempData = downsampler(data)
        tempData = tempData.drop(tempData.columns[0], axis=1)
        tempData = tempData.to_numpy()
        tempData = noiseRemover(tempData)
        finalData = (tempData - mean) / sigma   
        finalData = np.expand_dims(finalData, axis = -1)
        finalData = createMissingLeads(finalData)
        for i in range(finalData.shape[2]):
            np.save(f'/home/tzikos/Desktop/Data/Berts torch/{segment}/{usage}/{file.split("/")[-3]}-{file.split("/")[-1][:-5]}-{i}.npy', np.expand_dims(finalData[:, :, i], axis =-1))

