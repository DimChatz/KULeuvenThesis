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
            if file.endswith('.xlsx') and (segment in root) and ("overzicht" not in file):
                #print(file)
                excelFiles.append(os.path.join(root, file))
    #print(excelFiles)
    return excelFiles

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

def gaussianNormalizer(gaussianArray):
    gaussianArray = gaussianArray[:, :, 1:]
    mean = np.mean(gaussianArray)
    sigma = np.std(gaussianArray)
    gaussianArray = (gaussianArray - mean) / sigma
    return gaussianArray
                
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

# Main
folderPath = '/home/tzikos/Desktop/Data/Berts'
segmentList = ['AVNRT pre', 'AVNRT tachy', 'AVRT pre', 'AVRT tachy', 'concealed pre', 'concealed tachy', 'EAT pre', 'EAT tachy']
gaussianArray = np.zeros((2500,12,1), np.float64)
countList = []
for segment in segmentList:
    count = 0
    excelFiles = appendExcelFiles(folderPath, segment)
    for excelFile in excelFiles:
        count += 1
        tempData = pd.read_excel(excelFile)
        tempData = downsampler(tempData)
        tempData = tempData.drop(tempData.columns[0], axis=1)
        tempData = tempData.to_numpy()
        tempData = noiseRemover(tempData)
        tempExpanded = np.expand_dims(tempData, axis = -1)
        gaussianArray = np.concatenate((gaussianArray, tempExpanded), axis=2)
    countList.append(count)
finalData = gaussianNormalizer(gaussianArray)
for i in range(len(segmentList)):
    print(f'The segment is {segmentList[i]} and the size is {countList[i]}')
    saveData = finalData[:, :, :countList[i]]
    saveData = createMissingLeads(saveData)
    np.save(f'/home/tzikos/Desktop/Data/Berts torch/{segmentList[i]}.npy', saveData)
    finalData = finalData[:, :, countList[i]-1:]




