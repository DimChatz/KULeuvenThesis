import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")


def resampler(data, curFrequency):
    '''Function to adjust data frequency:
    - If data is 1000Hz, keep only the instance furthest from the mean in each group.
    - If data is 500Hz, return data as is.
    - If data is 250Hz, interpolate the mean between consecutive points.
    `data` is a numpy array where the first column is assumed to be time.
    '''
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
        downsampledData = data.copy()
    return downsampledData


def bandpass(lowF = 0.5, highF = 47, samplingF = 500, order = 5):
    '''Buttersworth bandpass filter'''
    # Nyquist frequency
    nyqF = 0.5 * samplingF
    # Cutoffs
    lowCut = lowF / nyqF
    highCut = highF / nyqF
    b, a = butter(order, [lowCut, highCut], btype='bandpass')
    return b, a


def noiseRemover(data, lowF = 0.5, highF = 47, samplingF = 500, order = 5):
    '''Application of bandpass'''
    b, a = bandpass(lowF, highF, samplingF, order)
    y = filtfilt(b, a, data, axis = 0)
    return y


def calcClassWeights(segmentList):
    experiment = segmentList[0].split(" ")[-1]
    '''Function to calculate the class weights for Berts data'''
    classInstance = []
    for i in range(len(segmentList)):
        classType = segmentList[i].split(" ")[0]
        classTypeSearch = f"-{classType}-"
        fileList = os.listdir(f"/home/tzikos/Desktop/Data/Berts final balanced augs/{experiment}/fold1/")
        tempList = [s for s in fileList if classTypeSearch in s]
        count = len(tempList)
        classInstance.append(count)
    totalInstances = int(sum(classInstance))
    classWeight = [totalInstances / (len(segmentList) * classInstance[i]) for i in range(len(segmentList))]
    print(f"class weights are: {classWeight}")
    np.save(f'/home/tzikos/Desktop/weights/{experiment}ClassWeights.npy', classWeight)


def originalNBaseNPs(segmentList):
    experiment = segmentList[0].split(" ")[-1]
    '''Function to create the original and base datasets for Berts data'''
    # Create the directories
    os.makedirs(f"/home/tzikos/Desktop/Data/Berts orig/{experiment}/", exist_ok=True)
    os.makedirs(f"/home/tzikos/Desktop/Data/Berts denoised/{experiment}/", exist_ok=True)
    # For each type of class
    for i in range(len(segmentList)):
        classType = segmentList[i].split(" ")[0]
        # Get the files
        for root, dirs, files in os.walk(f"/home/tzikos/Desktop/Data/Berts/{classType}/{segmentList[i]}/{segmentList[i]}"):
            for j, file in enumerate(files):
                # Get the correct only excel files
                if file.endswith(".xlsx") and "overzicht" not in file:
                    tempDF = pd.read_excel(f"{root}/{file}")
                    # Get the frequency
                    curFrequency = round(1 / (tempDF[tempDF.columns[0]].iloc[1] - tempDF[tempDF.columns[0]].iloc[0]))
                    # Transform dfs to np arrays
                    tempNP = tempDF.iloc[:, 1:].to_numpy()
                    tempNPDenoised = tempNP.copy()
                    # Save the original and base datasets
                    tempNPOrig = resampler(tempNP, curFrequency)
                    np.save(f"/home/tzikos/Desktop/Data/Berts orig/{experiment}/{classType}-patient{j}.npy", tempNPOrig)
                    tempNPDenoised = noiseRemover(tempNPDenoised, samplingF=curFrequency)
                    tempNPDenoised = resampler(tempNPDenoised, curFrequency)
                    np.save(f"/home/tzikos/Desktop/Data/Berts denoised/{experiment}/denoised-{classType}-patient{j}.npy", tempNPDenoised)

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
        fileList = [s for s in os.listdir(f"/home/tzikos/Desktop/Data/Berts orig/{experiment}") if classType in s]
        for j, file in enumerate(fileList):
            # Check that excel files are picked, 
            # they are not the general files for all the tests and
            idx = j % 10
            foldFiles[idx].append(os.path.join(f"/home/tzikos/Desktop/Data/Berts orig/{experiment}", file))
    for i in range(len(segmentList)):
        classType = segmentList[i].split(" ")[0]
        fileList = [s for s in os.listdir(f"/home/tzikos/Desktop/Data/Berts denoised/{experiment}") if classType in s]
        for j, file in enumerate(fileList):
            # Check that excel files are picked, 
            # they are not the general files for all the tests and
            idx = j % 10
            foldFiles[idx].append(os.path.join(f"/home/tzikos/Desktop/Data/Berts denoised/{experiment}", file))
    return foldFiles


def calcMeanSigma(directory, experiment):
    '''Function to create balanced datasets'''
    # For all classes
    toCalc = np.zeros((5000,12))
    for root, dirs, files in os.walk(f"{directory}"):
        for file in files:
            # Check that excel files are picked, 
            # they are not the general files for all the tests and
           if file.endswith('.npy') and "normal" in file:
                npArray = np.load(f"{directory}/{file}")
                toCalc = np.concatenate((toCalc, npArray), axis = 0)
    toCalc = toCalc[5000:,:]
    mean = np.mean(toCalc, axis=0)
    sigma = np.std(toCalc, axis=0)
    print(f"mean is {mean[0]:.4f}")
    print(f"sigma is {sigma[0]:.2f}")
    np.save(f'/home/tzikos/Desktop/weights/mean{experiment}Berts.npy', mean)
    np.save(f'/home/tzikos/Desktop/weights/sigma{experiment}Berts.npy', sigma)


def processorBert(foldList, segmentList, balanced=False):
    experiment = segmentList[0].split(" ")[-1]
    mean = np.load(f"/home/tzikos/Desktop/weights/mean{experiment}Berts.npy")
    sigma = np.load(f"/home/tzikos/Desktop/weights/mean{experiment}Berts.npy")
    if balanced:
        timesList = getTimesList(foldList[0], segmentList)
    else:
        timesList = np.ones((5), dtype=int)
    for i, fold in enumerate(foldList):
        for j, segment in enumerate(segmentList):
            classType = segment.split(" ")[0]
            filteredFold = [s for s in fold if classType in s]
            for file in filteredFold:
                tempNP = np.load(file)
                tempNP = (tempNP - mean) / sigma
                np.save(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/fold{i+1}/normalized-{file.split("/")[-1]}", tempNP)
                if timesList[j] > 0:
                    tempNPAug = augGaussianData(file, tempNP, timesList[j], experiment, i+1)
                    createMissingLeads(file, experiment, tempNPAug, i+1)
                createMissingLeads(file, experiment, tempNP, i+1)


def getTimesList(fileList, segmentList):
    experiment = segmentList[0].split(" ")[-1]
    countList = []
    for segment in segmentList:
        classType = segment.split(" ")[0]
        tempList = [s for s in fileList if classType in s]
        countList.append(len(tempList))
    denom = max(countList)
    timesList = [ int(denom / countList[i] - 1) for i in range(len(countList))]
    print(timesList)
    return timesList


def createMissingLeads(file, experiment, startArray, foldNum):
    '''Data Augmentation by removing one lead at a time''' 
    scaleList = [0.12, 0.08, 0.15, 0.2, 0.05, 0.02]
    augment = True
    if len(startArray.shape) < 3:
        augment = False
        startArray = np.expand_dims(startArray, axis=-1)
    for i in range (startArray.shape[2]):
        # For every column
        for j in range(startArray.shape[1]):
            # Copy the array...
            tempArray = startArray[:,:,i].copy()
            # ...to set the copy to 0...
            tempArray[:, j, ] = 0.
            # ...and save it
            if augment:
                np.save(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/fold{foldNum}/augmented{scaleList[i]}-missingLead{j+1}-{file.split("/")[-1]}", tempArray)       
            else:
                np.save(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/fold{foldNum}/missingLead{j+1}-{file.split("/")[-1]}", tempArray)


def augGaussianData(file, data, times, experiment, foldNum, independent=True):
    scaleList = [0.12, 0.08, 0.15, 0.2, 0.05, 0.02]
    noiseList = [0.2, 0.4/3, 0.25, 1/3, 0.2, 1/30]
    appendNP = np.zeros((data.shape[0], data.shape[1], 1))
    if times > 6:
        times = 6
    for i in range(times):
        if independent:
            gaussianScaler = np.random.normal(loc=1, scale=scaleList[i], size=data.shape)
        else:
            gaussianScaler = np.random.normal(loc=1, scale=scaleList[i], size=(data.shape[0], 1))
        gaussianNoise = np.random.normal(loc=0, scale=noiseList[i], size=data.shape)        
        augData = data * gaussianScaler + gaussianNoise
        np.save(f"/home/tzikos/Desktop/Data/Berts final/{experiment}/fold{foldNum}/augmented{scaleList[i]}-{file.split("/")[-1]}", augData)
        augData = np.expand_dims(augData, axis=-1)
        appendNP = np.concatenate((appendNP, augData), axis=-1)
    appendNP = appendNP[:,:,1:]
    return appendNP