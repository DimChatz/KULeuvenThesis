import os
import numpy as np

def classInstanceCalc(rootDir, targetDir,  segmentList, experiment):
    countList = []
    for i in range(len(segmentList)):
        count = 0
        for root, dirs, files in os.walk(rootDir):
            for file in files:
                if file.endswith('.xlsx') and ("overzicht" not in file) and (segmentList[i] in os.path.join(root, file)):
                    count += 1
        print(f'For segment {segmentList[i]} the total class count in training is {int(np.round(0.8*count))}')
        countList.append(count)
    countList = np.array(countList, dtype=np.float32)
    np.save(f'{targetDir}/{experiment}Weights.npy', countList)

def calcWeights(dir, experiment):
    classCounts = np.load(f'{dir}/{experiment}Weights.npy')
    totalCounts = np.sum(classCounts)
    classWeights = totalCounts / classCounts
    return classWeights


def countDigit(n):
    count = 0
    while n != 0:
        n //= 10
        count += 1
    return count