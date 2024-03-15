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


def bandpass(lowF = 0.5, highF = 45, samplingF = 250, order = 5):
    '''Buttersworth bandpass filter'''
    # Nyquist frequency
    nyqF = 0.5 * samplingF
    # Cutoffs
    lowCut = lowF / nyqF
    highCut = highF / nyqF
    b, a = butter(order, [lowCut, highCut], btype='bandpass')
    return b, a


def noiseRemover(data, lowF = 0.5, highF = 45, samplingF = 250, order = 5):
    '''Application of bandpass'''
    b, a = bandpass(lowF, highF, samplingF, order)
    y = filtfilt(b, a, data, axis = 0)
    return y


def appendStratifiedFiles(directory, segmentList):
    '''Function to create balanced datasets'''
    # Split trackers
    trainFiles, valFiles, testFiles = [], [], []
    # Class weight tracker
    classWeights = []
    # For all classes
    for i in range(len(segmentList)):
        excelFiles = []
        for root, dirs, files in os.walk(f"{directory}{segmentList[i].split(" ")[0]}/{segmentList[i]}/{segmentList[i]}"):
            for file in files:
                # Check that excel files are picked, 
                # they are not the general files for all the tests and 
                if file.endswith('.xlsx') and ("overzicht" not in file):
                    excelFiles.append(os.path.join(root, file))
        # Split 80-10-10
        print(f"The class {segmentList[i]} has {len(excelFiles)} files")
        classWeights.append(len(excelFiles))
        trainFiles += excelFiles[:int(np.round(0.8*len(excelFiles)))]
        valFiles += excelFiles[int(np.round(0.8*len(excelFiles))):int(np.round(0.9*len(excelFiles)))]
        testFiles += excelFiles[int(np.round(0.9*len(excelFiles))):]
    # Create , print and save class weights
    classWeights = np.array(classWeights, dtype=np.float32)
    classWeights = np.sum(classWeights) / (classWeights * len(segmentList))
    print(f"The class weights for Bert are of shape {classWeights.shape} and are {classWeights}")
    np.save(f'/home/tzikos/Desktop/weights/Bert{segmentList[i].split(" ")[1]}Weights.npy', classWeights)
    return trainFiles, valFiles, testFiles


def downsamplerNoiseRemover(listToSave, targetDirDown, targetDirNoise):
    '''Function to downsample and remove noise'''
    os.makedirs(targetDirDown, exist_ok=True)
    os.makedirs(targetDirNoise, exist_ok=True)
    for file in listToSave:
        tempData = pd.read_excel(f"{file}")
        tempData = downsampler(tempData)
        tempData = tempData.drop(tempData.columns[0], axis=1)
        tempArray = tempData.to_numpy()
        np.save(f"{targetDirDown}/{file.split('/')[-2]}-{file.split('/')[-1][:-5]}.npy", tempArray)
        tempArray = noiseRemover(tempArray)
        np.save(f"{targetDirNoise}/{file.split('/')[-2]}-{file.split('/')[-1][:-5]}.npy", tempArray)


def gaussianCalcBert(directory):
    '''Function to calculate the mean and sigma for Bert's data'''
    meanAcc, sigmaAcc = np.zeros((1,12)), np.zeros((1,12))
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                meanAcc = np.concatenate((meanAcc, np.expand_dims(np.mean(tempArray, axis=0), axis=0)), axis = 0)
                sigmaAcc = np.concatenate((sigmaAcc, np.expand_dims(np.mean(tempArray, axis=0), axis=0)), axis = 0)
    # Remove zero duplicate at start
    meanAcc, sigmaAcc = meanAcc[1:], sigmaAcc[1:]
    # Create mean and sigma
    mean = np.sum(meanAcc, axis = 0) / meanAcc.shape[0]
    sigmaSum = np.sum(np.power(sigmaAcc, 2) + np.power(meanAcc, 2), axis = 0)
    sigma = np.sqrt((sigmaSum - meanAcc.shape[0] * np.power(mean, 2)) / meanAcc.shape[0])
    np.save(f'/home/tzikos/Desktop/weights/meanBerts.npy', mean)
    np.save(f'/home/tzikos/Desktop/weights/sigmaBerts.npy', sigma)
    return mean, sigma


def gaussianNormalizerMissingLeadCreatorBert(rootDir, targetDirGaussian, targetDirMissing, 
                                             mean, sigma):
    '''Function to normalize and create missing leads for Bert's data'''
    os.makedirs(targetDirGaussian, exist_ok=True)
    os.makedirs(targetDirMissing, exist_ok=True)
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                tempArray = (tempArray - mean) / sigma
                np.save(f"{targetDirGaussian}/{file}", tempArray)
                tempArray = createMissingLeads(targetDirMissing, file, tempArray)


def createMissingLeads(targetDir, file, startArray):
    '''Data Augmentation by removing one lead at a time'''
    endArray = startArray.copy()
    # For every column
    for i in range(startArray.shape[1]):
        # Copy the array...
        tempArray = startArray.copy()
        # ...to set the copy to 0...
        tempArray[:, i] = 0.
        # ...and save it
        np.save(f"{targetDir}/{file.split('/')[-1][:-4]}-{i+1}.npy", tempArray)
    np.save(f"{targetDir}/{file.split('/')[-1][:-4]}-{0}.npy", endArray)
    return endArray