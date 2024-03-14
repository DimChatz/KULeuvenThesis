import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")



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
                    print("NaN found")

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
                    print(f"Duplicates found in geadnr in {file}")
                    print(duplicates)

