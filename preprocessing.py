import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")

def downsampler(data):
    '''Function for downsampling all data to 250Hz.'''
    # Determine the current sampling frequency
    current_frequency = round(1 / data[data.columns[0]].iloc[1] - data[data.columns[0]].iloc[0])
    # Downsample if necessary
    if current_frequency == 500:
        # Take every other row to achieve 250Hz from 500Hz
        data = data.iloc[::2].reset_index(drop=True)
        # Take every 4th row to achieve 250Hz from 1kHz
    elif current_frequency == 1000:
        data = data.iloc[::4].reset_index(drop=True)
    return data

def appendExcelFiles(directory, segment):
    '''Function to recursively search in all files
        to be used in later normalization'''
    excelFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check that excel files are picked, 
            # they are not the general files for all the tests and 
            # that they are of the correct experiment type (pre, tachy)
            if file.endswith('.xlsx') and ("overzicht" not in file) and (segment in os.path.join(root, file)):
                excelFiles.append(os.path.join(root, file))
    return excelFiles

def appendStratifiedFiles(directory, segmentList):
    '''Function to create balanced datasets'''
    trainFiles, valFiles, testFiles = [], [], []
    # For all classes
    for i in range(len(segmentList)):
        excelFiles = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Check that excel files are picked, 
                # they are not the general files for all the tests and 
                # that they are of the correct experiment type (pre, tachy) 
                if file.endswith('.xlsx') and ("overzicht" not in file) and (segmentList[i] in os.path.join(root, file)):
                    excelFiles.append(os.path.join(root, file))
        # Split 80-10-10
        trainFiles += excelFiles[:int(np.round(0.8*len(excelFiles)))]
        valFiles += excelFiles[int(np.round(0.8*len(excelFiles))):int(np.round(0.9*len(excelFiles)))]
        testFiles += excelFiles[int(np.round(0.9*len(excelFiles))):]
    return trainFiles, valFiles, testFiles

def gaussianCalc(gaussianArray):
    '''Calcs mean and sigma for Gaussian'''
    mean, sigma = [], []
    for i in range(12):
        mean.append(np.mean(gaussianArray[:,i]))
        sigma.append(np.std(gaussianArray[:,i]))
    return mean, sigma

def gaussianNormalizer(folderPath, segment):
    '''Apply Gaussian Normalization'''
    gaussianArray = np.zeros((2500,12), np.float32)
    excelFiles = appendExcelFiles(folderPath, segment)
    for excelFile in excelFiles:
        try:
            tempData = pd.read_excel(excelFile)
            tempData = downsampler(tempData)
            tempData = tempData.drop(tempData.columns[0], axis=1)
            tempData = tempData.to_numpy()
            # Apply noise removal filter - bandpass 0.5-45Hz
            tempData = noiseRemover(tempData)
            gaussianArray = np.vstack((gaussianArray, tempData))
        except Exception as e:
            print(excelFile)
            print(e)
    return gaussianCalc(gaussianArray)
    
def createMissingLeads(startArray):
    '''Data Augmentation by removing one lead at a time'''
    endArray = startArray
    # For every column
    for i in range(startArray.shape[1]):
        # Copy the array...
        tempArray = startArray.copy()
        # ...to set the copy to 0...
        tempArray[:, i, :] = 0.
        # ...and add it back
        endArray = np.concatenate((endArray, tempArray), axis = 2)
    return endArray

def bandpass(lowF = 0.5, highF = 45, samplingF = 250, order = 5):
    '''Buttersworth bandpass filter'''
    nyqF = 0.5 * samplingF
    lowCut = lowF / nyqF
    highCut = highF / nyqF
    b, a = butter(order, [lowCut, highCut], btype='bandpass')
    return b, a

def noiseRemover(data, lowF = 0.5, highF = 45, samplingF = 250, order = 5):
    '''Application of bandpass'''
    b, a = bandpass(lowF, highF, samplingF, order)
    y = filtfilt(b, a, data, axis = 0)
    return y

def preprocPipeline(files, usage, segment, mean, sigma):
    '''Final processing pipeline'''
    for file in files:
        data = pd.read_excel(file)
        # Make it all 250Hz
        tempData = downsampler(data)
        # Drop time axis
        tempData = tempData.drop(tempData.columns[0], axis=1)
        tempData = tempData.to_numpy()
        # Remove noise
        tempData = noiseRemover(tempData)
        # Normalize to calculated Gaussian
        finalData = np.zeros((2500, 12), np.float32)
        for i in range(12):
            finalData[:, i] = (tempData[:, i] - mean[i]) / sigma[i]   
        finalData = np.expand_dims(finalData, axis = -1)
        # Create extra leads
        finalData = createMissingLeads(finalData)
        for i in range(finalData.shape[2]):
            # Save so you don't do all the above every time
            np.save(f'/home/tzikos/Desktop/Data/Berts torch/{segment}/{usage}/{file.split("/")[-3]}-{file.split("/")[-1][:-5]}-{i}.npy', np.expand_dims(finalData[:, :, i], axis =-1))

def preprocessAll(folderpath, segmentList, mean, sigma):
    '''Final function
        Applies the above as needed'''
    # Create train, val, test
    trainData, valData, testData = appendStratifiedFiles(folderpath, segmentList)
    print(f"Train Data is, {len(trainData)}, val is {len(valData)}, test is {len(testData)}")
    # Get type of experiment
    experiment = segmentList[0].split(" ")[-1]
    # Preproc
    preprocPipeline(trainData, "train", experiment, mean, sigma)
    preprocPipeline(valData, "val", experiment, mean, sigma)
    preprocPipeline(testData, "test", experiment, mean, sigma  )