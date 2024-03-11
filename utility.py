import os
import numpy as np
import torch
import torch.nn.functional as F

def padSeqSymm(batch, targetLength, dimension):
    """
    Symmetrically pad the sequences in a batch to have a uniform length.
    
    Parameters:
    - batch: A batch of input data of shape (batch_size, channels, sequence_length)
    - target_length: The desired length for all sequences.
    - dimension: Dimension along which to pad. For 1D convolution on sequences, it is usually 2.
    
    Returns:
    - Symmetrically padded batch of input data.
    """
    currentLength = batch.size(dimension)
    totPadNeeded = max(0, targetLength - currentLength)
    # Calculate padding to add to both the start and end of the sequence
    padEachSide = totPadNeeded // 2
    # For an odd total_padding_needed, add the extra padding at the end
    if totPadNeeded % 2 == 1:
        padArg = [padEachSide, padEachSide + 1] + [0, 0] * (batch.dim() - dimension - 1)
    else:
        padArg = [padEachSide, padEachSide] + [0, 0] * (batch.dim() - dimension - 1)
    symPadBatch = F.pad(batch, pad=padArg, mode='constant', value=0)
    return symPadBatch

def classInstanceCalc(rootDir, targetDir,  segmentList, experiment):
    os.makedirs(targetDir, exist_ok=True)
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
    print(f'Saves at {targetDir}{experiment}Weights.npy')
    np.save(f'{targetDir}{experiment}Weights.npy', countList)

def calcWeights(dir, experiment):
    print(f'Saved at {dir}{experiment}Weights.npy')
    classCounts = np.load(f'{dir}{experiment}Weights.npy')
    totalCounts = np.sum(classCounts)
    classWeights = totalCounts / classCounts
    return classWeights


def countDigit(n):
    count = 0
    while n != 0:
        n //= 10
        count += 1
    return count