import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Workbook contains no default style, apply openpyxl's default")

# Function to recursively search for Excel files in a directory
def findExcelFiles(directory, segment):
    excelFiles = []
    countTotal = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx') and (segment in root) and ("overzicht" not in file):
                excelFiles.append(os.path.join(root, file))
                countTotal += 1
    return excelFiles, countTotal

# Function to check for type of data
def checkColumnDiscernibility(excelFile, countNew):
    df = pd.read_excel(excelFile)
    if (df.iloc[:,1:].head().astype(float) % 2.5 !=0).all().all():
        countNew += 1
    return countNew

# Main
folderPath = '/home/tzikos/Desktop/Data/'
segmentList = ['AVNRT pre', 'AVNRT tachy', 'AVRT pre', 'AVRT tachy', 'concealed pre', 'concealed tachy', 'EAT pre', 'EAT tachy']
for segment in segmentList:
    excelFiles, countTotal = findExcelFiles(folderPath, segment)
    countNew = 0
    for excelFile in excelFiles:
        countNew = checkColumnDiscernibility(excelFile, countNew)
    print(f"For the case of {segment} we have {countTotal - countNew} old ECGs and {countNew} new")