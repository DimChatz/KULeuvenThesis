import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")
from tqdm import tqdm
from functools import reduce


def resampler(data, curFrequency):
    '''Function to adjust data frequency:
    - If data is 1000Hz, keep only the instance furthest from the mean in each group.
    - If data is 500Hz, return data as is.
    - If data is 250Hz, interpolate the mean between consecutive points.
    `data` is a numpy array where the first column is assumed to be time.'''
    # Remove the time column for processing
    if curFrequency == 1000:
        downsampledData = data[::2, :]
    elif curFrequency == 500:
        # Data is already at desired frequency
        downsampledData = data.copy()
    elif curFrequency == 250:
        # Interpolate by calculating the mean with the next data point
        downsampledData = np.empty((data.shape[0]*2, data.shape[1]))
        for i in range(data.shape[0]-1):
            downsampledData[2*i, :] = data[i, :]
            interpolatedRow = (data[i, :] + data[i+1, :]) / 2
            downsampledData[2*i+1, :] = interpolatedRow
        downsampledData[-2, :] = data[-1, :]
        downsampledData[-1, :] = data[-1, :]
    else:
        # Frequency not recognized, return data as is
        raise ValueError("Frequency not recognized")
    return downsampledData


def bandpass(lowF = 0.5, highF = 49, samplingF = 500, order = 5):
    '''Buttersworth bandpass filter'''
    # Nyquist frequency
    nyqF = 0.5 * samplingF
    # Cutoffs
    lowCut = lowF / nyqF
    highCut = highF / nyqF
    b, a = butter(order, [lowCut, highCut], btype='bandpass')
    return b, a


def noiseRemover(data, lowF = 0.5, highF = 49, samplingF = 500, order = 5):
    '''Application of bandpass'''
    b, a = bandpass(lowF, highF, samplingF, order)
    y = filtfilt(b, a, data, axis = 0)
    return y


def calcClassWeights(segmentList):
    experiment = segmentList[0].split(" ")[-1]
    '''Function to calculate the class weights for Berts data'''
    classInstance = []
    for i in tqdm(range(len(segmentList))):
        classType = segmentList[i].split(" ")[0]
        classTypeSearch = f"-{classType}-"
        fileList = os.listdir(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/fold1/")
        tempList = [s for s in fileList if classTypeSearch in s]
        count = len(tempList)
        classInstance.append(count)
    totalInstances = int(sum(classInstance))
    binaryInstances = int(sum(classInstance[1:]))
    classWeight = [totalInstances / (len(segmentList) * classInstance[i]) for i in range(len(segmentList))]
    print(f"class weights are: {classWeight}")
    AVRTTotal = classInstance[1] + classInstance[2] + classInstance[3]
    binaryClassWeight = [totalInstances / 2 / classInstance[0],  totalInstances / 2 / binaryInstances]
    AVRTClassWeight = [AVRTTotal / 2 / classInstance[1],  AVRTTotal / 2 / (classInstance[2]+classInstance[3])]
    np.save(f'/home/tzikos/Desktop/weights/{experiment}ClassWeights.npy', classWeight)
    np.save(f'/home/tzikos/Desktop/weights/{experiment}ClassWeightsBinary.npy', binaryClassWeight)
    np.save(f'/home/tzikos/Desktop/weights/{experiment}ClassWeightsAVRT.npy', AVRTClassWeight)


def originalNBaseNPs(segmentList):
    experiment = segmentList[0].split(" ")[-1]
    os.makedirs(f"/home/tzikos/Desktop/Data/Berts orig/{experiment}/", exist_ok=True)
    '''Function to create the original and base datasets for Berts data'''
    # Create the directories
    # For each type of class
    for i in range(len(segmentList)):
        classType = segmentList[i].split(" ")[0]
        # Get the files
        fileList = os.listdir(f"/home/tzikos/Desktop/Data/Berts/{classType}/{segmentList[i]}/{segmentList[i]}")
        for file in tqdm(fileList):
            # Get the correct only excel files
            if file.endswith(".xlsx") and "overzicht" not in file:
                tempDF = pd.read_excel(f"/home/tzikos/Desktop/Data/Berts/{classType}/{segmentList[i]}/{segmentList[i]}/{file}")
                # Get the frequency
                curFrequency = round(1 / (tempDF[tempDF.columns[0]].iloc[1] - tempDF[tempDF.columns[0]].iloc[0]))
                # Transform dfs to np arrays
                tempNP = tempDF.iloc[:, 1:].to_numpy()
                # Save the original and denoised datasets
                tempNPDenoised = noiseRemover(tempNP, highF=100, samplingF=curFrequency)
                tempNPDenoised = resampler(tempNPDenoised, curFrequency)
                np.save(f"/home/tzikos/Desktop/Data/Berts orig/{experiment}/orig-{classType}-{file.split(".")[-2]}.npy", tempNPDenoised)


def foldCreator(segmentList):
    experiment = segmentList[0].split(" ")[-1]
    os.makedirs(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/", exist_ok=True)
    '''Function to create balanced datasets'''
    # For all classes
    foldFiles = [[] for _ in range(10)]
    for j in range(10):
        os.makedirs(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/fold{j+1}", exist_ok=True)
    for i in range(len(segmentList)):
        classType = segmentList[i].split(" ")[0]
        fileList = [file for file in os.listdir(f"/home/tzikos/Desktop/Data/Berts orig/{experiment}") if classType in file]
        print(f"Class {classType} has {len(fileList)} files")
        for j, file in tqdm(enumerate(fileList)):
            # Check that excel files are picked, 
            # they are not the general files for all the tests and
            idx = j % 10
            foldFiles[idx].append(os.path.join(f"/home/tzikos/Desktop/Data/Berts orig/{experiment}", file))
    return foldFiles


def calcMeanSigma(fileList, experiment):
    '''Function to create balanced datasets'''
    flatList = [item for sublist in fileList for item in sublist]
    normList = [file for file in flatList if "normal" in file]
    nonNormList = [file for file in flatList if "normal" not in file]
    nonFlatList = [normList, nonNormList]
    flatFileList = [item for sublist in nonFlatList for item in sublist]
    meanArray = np.zeros(12, dtype=np.float64)
    sigmaArray = np.zeros(12, dtype=np.float64)
    for i in range(12):
        # For all classes
        toCalc = np.zeros((5000,1))
        for file in tqdm(flatFileList):
            npArray = np.expand_dims(np.load(file)[:,i], axis=-1)
            if np.max(npArray) > 1e5:
                print(f"File {file} has a max of {np.max(npArray)}")
            toCalc = np.concatenate((toCalc, npArray), axis = 1)
            # Remove the first duplicate of zeros
        toCalc = toCalc[:,1:]
        print(toCalc.shape)
        # Calculate the mean and sigma
        mean = np.mean(toCalc)
        print(mean)
        sigma = np.std(toCalc)
        print(sigma)
        meanArray[i] = mean
        sigmaArray[i] = sigma
    print(f"means are {meanArray}")
    print(f"sigmas are {sigmaArray}")
    np.save(f'/home/tzikos/Desktop/weights/mean{experiment}Berts.npy', meanArray)
    np.save(f'/home/tzikos/Desktop/weights/sigma{experiment}Berts.npy', sigmaArray)


def processorBert(foldList, segmentList):
    experiment = segmentList[0].split(" ")[-1]
    mean = np.load(f"/home/tzikos/Desktop/weights/mean{experiment}Berts.npy")
    sigma = np.load(f"/home/tzikos/Desktop/weights/mean{experiment}Berts.npy")
    for i, fold in tqdm(enumerate(foldList)):
        for file in fold:
            tempNP = np.load(file)
            tempNP = (tempNP - mean) / sigma
            np.save(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/fold{i+1}/{file.split("/")[-1]}", tempNP)
            createMissingLeads(file, experiment, tempNP, i+1)


def createMissingLeads(file, experiment, startArray, foldNum):
    '''Data Augmentation by removing one lead at a time'''
    # For every column
    for j in range(startArray.shape[1]):
        # Copy the array...
        tempArray = startArray.copy()
        # ...to set the copy to 0...
        tempArray[:, j] = 0.
        # ...and save it
        np.save(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/fold{foldNum}/missingLead{j+1}-{file.split("/")[-1]}", tempArray)