import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")
from preprocessing import appendStratifiedFiles, downsampler


def classInstanceCalc(rootDir, targetDir,  segmentList, experiment):
    """Function to calculate the class instances in Bert's training set
    Params:
    - rootDir: the root directory of the data
    - targetDir: the directory to save the class instances
    - segmentList: the list of class names
    - experiment: the experiment type (pre or tachy)"""

    # Create directory if it does not exist to save the data
    os.makedirs(targetDir, exist_ok=True)
    # Initialize the list to save the class instances
    countList = []
    # For all segments
    for i in range(len(segmentList)):
        # Initialize the count
        count = 0
        # Recursively search for excel files in the directory
        for root, dirs, files in os.walk(rootDir):
            for file in files:
                # Check that excel files are picked, 
                # they are not the general files for all the tests and 
                # that they are of the correct experiment type (pre, tachy) 
                if file.endswith('.xlsx') and ("overzicht" not in file) and (segmentList[i] in os.path.join(root, file)):
                    # Add to the count
                    count += 1
        print(f'For segment {segmentList[i]} the total class count in training is {int(np.round(0.8*count))}')
        countList.append(count)
    # Save the class instances
    countList = np.array(countList, dtype=np.float32)
    np.save(f'{targetDir}{experiment}Weights.npy', countList)

def calcWeights(dir, experiment, classes):
    """Function to calculate the class weights"""
    # Load the class instances
    classCounts = np.load(f'{dir}{experiment}Weights.npy')
    # Calculate the total counts
    totalCounts = np.sum(classCounts)
    # Calculate the class weights
    classWeights = totalCounts / (classCounts * classes)
    print(f'Class weights are {classWeights}')
    return classWeights


def countDigit(n):
    """Function to count the number of digits in a number"""
    count = 0
    while n != 0:
        n //= 10
        count += 1
    return count

def checkBertMissing(directory):
    """Function to check for missing leads in Bert's data"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check that excel files are picked, 
            # they are not the general files for all the tests and 
            # that they are of the correct experiment type (pre, tachy)
            if file.endswith('.xlsx') and ("overzicht" not in file):
                df = pd.read_excel(os.path.join(root, file))
                # Check for missing data in ecg leads
                if df.isna().any().any():
                    print(f"NaN found in file {file}")

def findDuplicatePatients(directory):
    """Function to find duplicate patients in the data"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check that excel files are picked, 
            # they are not the general files for all the tests and 
            # that they are of the correct experiment type (pre, tachy)
            if file.endswith('.xlsx') and ("overzicht" in file):
                df = pd.read_excel(os.path.join(root, file))
                # Check for duplicates in column 'geadnr'
                duplicates = df[df.duplicated(subset=['geadnr'], keep=False)]
                # Print the duplicates if any along with the file name
                if not duplicates.empty:
                    print(f"Duplicates found in geadnr in {file}")
                    print(duplicates)
                # Check for duplicates in column 'pseudo ID'
                duplicates = df[df.duplicated(subset=['pseudo ID'], keep=False)]
                # Print the duplicates if any along with the file name
                if not duplicates.empty:
                    print(f"Duplicates found in pseudo ID in {file}")
                    print(duplicates)

def checkStats(dataPath, exp):
    expList = [f"norm {exp}", f"AVNRT {exp}", f"AVRT {exp}", f"concealed {exp}", f"EAT {exp}"]
    meanTrain = np.load(f"/home/tzikos/Desktop/weights/meanBerts{exp}.npy")
    sigmaTrain = np.load(f"/home/tzikos/Desktop/weights/sigmaBerts{exp}.npy")
    accArray = np.zeros((2500,12,1))
    for root, dirs, files in os.walk(f"{dataPath}/{exp}/val"):
        for file in files:
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                accArray = np.concatenate((accArray, np.expand_dims(tempArray, axis=-1)), axis = 2)
    # Remove zero duplicate at start
    accArray = accArray[:,:,1:]
    # Create mean and sigma
    meanVal = np.mean(accArray, axis = (0,2))
    sigmaVal = np.std(accArray, axis = (0,2))

    accArray = np.zeros((2500,12,1))
    for root, dirs, files in os.walk(f"{dataPath}/{exp}/test"):
        for file in files:
            if file.endswith('.npy'):
                tempArray = np.load(os.path.join(root, file))
                accArray = np.concatenate((accArray, np.expand_dims(tempArray, axis=-1)), axis = 2)
    # Remove zero duplicate at start
    accArray = accArray[:,:,1:]
    # Create mean and sigma
    meanTest = np.mean(accArray, axis = (0,2))
    sigmaTest = np.std(accArray, axis = (0,2))

    with open(f"/home/tzikos/statsBert{exp}.txt", "w") as file:
        for i in range(5):
            file.write(f"Class {expList[i]}\n")
            for j in range(12):
                file.write(f"for lead {j}\n")
                file.write(f"For train the mean is {meanTrain[j]:.4f} and sigma {sigmaTrain[j]:.2f}\n")
                file.write(f"For val the mean is {meanVal[j]:.4f} and sigma {sigmaVal[j]:.2f}\n")
                file.write(f"For test the mean is {meanTest[j]:.4f} and sigma {sigmaTest[j]:.2f}\n")
                file.write("\n")
            file.write("\n")
            file.write("\n")


def tableCreator(expList):
    tableForVis = pd.DataFrame(columns = ['Dataset', 'Class', 'Lead', 'Mean', 'Sigma'])
    for i in range(len(expList)):
        excelFiles = []
        for root, dirs, files in os.walk(f"/home/tzikos/Desktop/Data/Berts/{expList[i].split(" ")[0]}/{expList[i]}/{expList[i]}"):
            for file in files:
                # Check that excel files are picked, 
                # they are not the general files for all the tests and 
                if file.endswith('.xlsx') and ("overzicht" not in file):
                    excelFiles.append(os.path.join(root, file))
        trainFiles = excelFiles[:int(0.8*len(excelFiles))]
        valFiles = excelFiles[int(0.8*len(excelFiles)):int(0.9*len(excelFiles))]
        testFiles = excelFiles[int(0.9*len(excelFiles)):]
        for file in trainFiles:
            tempData = pd.read_excel(f"{file}")
            tempData = downsampler(tempData)
            tempData = tempData.drop(tempData.columns[0], axis=1)
            for j in range(12):
                tableForVis.loc[len(tableForVis)] = ["Train", expList[i].split(" ")[0], j+1, np.mean(tempData[tempData.columns[j]]), np.std(tempData[tempData.columns[j]])]
        for file in valFiles:
            tempData = pd.read_excel(f"{file}")
            tempData = downsampler(tempData)
            tempData = tempData.drop(tempData.columns[0], axis=1)
            for j in range(12):
                tableForVis.loc[len(tableForVis)] = ["Val", expList[i].split(" ")[0], j+1, np.mean(tempData[tempData.columns[j]]), np.std(tempData[tempData.columns[j]])]
        for file in testFiles:
            tempData = pd.read_excel(f"{file}")
            tempData = downsampler(tempData)
            tempData = tempData.drop(tempData.columns[0], axis=1)
            for j in range(12):
                tableForVis.loc[len(tableForVis)] = ["Test", expList[i].split(" ")[0], j+1, np.mean(tempData[tempData.columns[j]]), np.std(tempData[tempData.columns[j]])]
    tableForVis.to_csv(f'/home/tzikos/TableCreatorVis/{expList[0].split(" ")[-1]}VisTable.csv', index=False)
