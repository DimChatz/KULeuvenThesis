import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import torch.nn.init as init
from itertools import chain
import math
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b

#############
### UTILS ###
#############
def padSeqSymm(batch, targetLength, dimension):
    """Symmetrically pad the sequences in a batch to have a uniform length.
    Params:
    - batch: A batch of input data of shape (batchSize, channels, sequenceLength)
    - targetLength: The desired length for all sequences.
    - dimension: Dimension along which to pad. For our task it's 2."""
    # Calculate the current length of the sequences
    currentLength = batch.size(dimension)
    # Calculate the total padding needed
    totPadNeeded = max(0, targetLength - currentLength)
    # Calculate padding to add to both the start and end of the sequence
    padEachSide = totPadNeeded // 2
    # For an odd padding needed, add the extra padding at the end (+1 at first list, second element)
    if totPadNeeded % 2 == 1:
        padArg = [padEachSide, padEachSide + 1] + [0, 0] * (batch.dim() - dimension - 1)
    # For an even padding needed, add the padding symmetrically
    else:
        padArg = [padEachSide, padEachSide] + [0, 0] * (batch.dim() - dimension - 1)
    # Pad the sequences
    symPadBatch = F.pad(batch, pad=padArg, mode='constant', value=0)
    return symPadBatch


def foldFinder(path, foldNum, ignoreMissing=True):
    for folder in os.listdir(path):
        if folder == f"fold{foldNum+1}":
            foldPath = os.path.join(path, folder)
            foldFiles = os.listdir(foldPath)
            foldFiles = [os.path.join(foldPath, file) for file in foldFiles]
            if ignoreMissing:
                foldFiles = [file for file in foldFiles if ("missing" not in file)]
    return foldFiles

def lengthFinder(path, valNum, norm_psvt=False):
    trainFilesList = []
    if norm_psvt:
        for folder in os.listdir(path):
            if folder == f"fold{valNum+1}":
                valPath = os.path.join(path, folder)
                valFiles = os.listdir(valPath)
                valFiles = [os.path.join(valPath, file) for file in valFiles]
                valFiles = [file for file in valFiles if (("missing" not in file) and ("AVRT" not in file))]
            elif folder == f"fold{(valNum+1)%10+1}":
                testPath = os.path.join(path, folder)
                testFiles = os.listdir(testPath)
                testFiles = [os.path.join(testPath, file) for file in testFiles]
                testFiles = [file for file in testFiles if (("missing" not in file) and ("AVRT" not in file))]
            else:
                trainPath = os.path.join(path, folder)
                trainFiles = os.listdir(trainPath)
                trainFiles = [os.path.join(trainPath, file) for file in trainFiles]
                trainFiles = [file for file in trainFiles if "AVRT" not in file]
                trainFiles = [file for file in trainFiles if "missing" not in file]
                trainFilesList.append(trainFiles)  
    else:
        for folder in os.listdir(path):
            if folder == f"fold{valNum+1}":
                valPath = os.path.join(path, folder)
                valFiles = os.listdir(valPath)
                valFiles = [os.path.join(valPath, file) for file in valFiles]
                valFiles = [file for file in valFiles if "missing" not in file]
            elif folder == f"fold{(valNum+1)%10+1}":
                testPath = os.path.join(path, folder)
                testFiles = os.listdir(testPath)
                testFiles = [os.path.join(testPath, file) for file in testFiles]
                testFiles = [file for file in testFiles if "missing" not in file]
            else:
                trainPath = os.path.join(path, folder)
                trainFiles = os.listdir(trainPath)
                trainFiles = [os.path.join(trainPath, file) for file in trainFiles]
                trainFiles = [file for file in trainFiles if "missing" not in file]
                trainFilesList.append(trainFiles)
    trainFilesList = list(chain.from_iterable(trainFilesList))
    return trainFilesList, valFiles, testFiles


def lengthFinderBinary(path, valNum):
    trainFilesList = []
    for folder in os.listdir(path):
        if folder == f"fold{valNum+1}":
            valPath = os.path.join(path, folder)
            valFiles = os.listdir(valPath)
            valFiles = [os.path.join(valPath, file) for file in valFiles]
            valFiles = [file for file in valFiles if ("missing" not in file) and (("AVNRT" in file) or ("AVRT" in file) or ("concealed" in file))]
        elif folder == f"fold{(valNum+1)%10+1}":
            testPath = os.path.join(path, folder)
            testFiles = os.listdir(testPath)
            testFiles = [os.path.join(testPath, file) for file in testFiles]
            testFiles = [file for file in testFiles if "missing" not in file and (("AVNRT" in file) or ("AVRT" in file) or ("concealed" in file))]
        else:
            trainPath = os.path.join(path, folder)
            trainFiles = os.listdir(trainPath)
            trainFiles = [os.path.join(trainPath, file) for file in trainFiles]
            trainFiles = [file for file in trainFiles if ((("AVNRT" in file) or ("AVRT" in file) or ("concealed" in file)))]
            #trainFiles = [file for file in trainFiles if "missing" not in file]
            trainFilesList.append(trainFiles)
    trainFilesList = list(chain.from_iterable(trainFilesList))
    return trainFilesList, valFiles, testFiles

################
### DATASETS ###
################
class ECGDataset2_0(torch.utils.data.Dataset):
    '''Time series Dataset'''
    def __init__(self, fileList, experiment, swin=False):
        # Root of data
        self.fileList = fileList
        #self.fileList = [file for file in self.fileList if "missing" not in file]
        # Type of classification task
        self.exp = experiment
        self.lengthData = len(self.fileList)
        self.swin = swin
        # Define the name dictionary
        self.nameDict = {f"normal {self.exp}": [1, 0, 0, 0, 0],
                         f"AVNRT {self.exp}": [0, 1, 0, 0, 0],
                         f"AVRT {self.exp}": [0, 0, 1, 0, 0],
                         f"concealed {self.exp}": [0, 0, 0, 1, 0],
                         f"EAT {self.exp}": [0, 0, 0, 0, 1]}

    def __len__(self):
        return self.lengthData

    def __getitem__(self, idx):
        # Get file
        inputData = np.load(self.fileList[idx]).astype(np.float32)
        # Get class label
        fileKey = self.fileList[idx].split("/")[-1].split("-")[-6]
        label = self.nameDict[f"{fileKey} {self.exp}"]
        # Reduce data shape (squeeze)
        # Transpose axis to proper format
        # and make it float32, not double
        inputData = torch.from_numpy(inputData)
        ecgData = inputData.transpose(1, 0)
        ecgLabels = np.array(label, dtype=np.float32)
        # Make them torch tensors
        ecgLabels = torch.from_numpy(ecgLabels)
        if self.swin:
            ecgData = ecgData.unsqueeze(0)
            ecgData = torch.cat([ecgData, ecgData, ecgData], dim=0)
        return ecgData, ecgLabels

#################################

class ECGDataset2_0Binary(torch.utils.data.Dataset):
    '''Time series Dataset'''
    def __init__(self, fileList, experiment, swin=False, AVRT=False):
        # Root of data
        self.fileList = fileList
        # Type of classification task
        self.exp = experiment
        self.lengthData = len(self.fileList)
        self.swin = swin
        # Define the name dictionary
        self.AVRT = AVRT
        if not self.AVRT:
            self.nameDict = {f"normal {self.exp}": [1, 0],
                            f"AVNRT {self.exp}": [0, 1],
                            f"AVRT {self.exp}": [0, 1],
                            f"concealed {self.exp}": [0, 1],
                            f"EAT {self.exp}": [0, 1]}
        else:
            self.nameDict = {f"AVNRT {self.exp}": [1, 0],
                            f"AVRT {self.exp}": [0, 1],
                            f"concealed {self.exp}": [0, 1]}
            
    def __len__(self):
        return self.lengthData

    def __getitem__(self, idx):
        # Get file
        inputData = np.load(self.fileList[idx]).astype(np.float32)
        # Get class label
        fileKey = self.fileList[idx].split("/")[-1].split("-")[-6]
        label = self.nameDict[f"{fileKey} {self.exp}"]
        # Reduce data shape (squeeze)
        # Transpose axis to proper format
        # and make it float32, not double
        inputData = torch.from_numpy(inputData)
        ecgData = inputData.transpose(1, 0)
        ecgLabels = np.array(label, dtype=np.float32)
        # Make them torch tensors
        ecgLabels = torch.from_numpy(ecgLabels)
        if self.swin:
            ecgData = ecgData.unsqueeze(0)
            ecgData = torch.cat([ecgData, ecgData, ecgData], dim=0)
        return ecgData, ecgLabels

#################################
    
class PTBDataset(torch.utils.data.Dataset):
    '''Time series Dataset'''
    def __init__(self, dirPath, numClasses, swin=False):
        # Root of data
        self.dir = dirPath
        # Its files
        self.fileNames = os.listdir(dirPath)
        self.fileNames = [os.path.join(self.dir, file) for file in self.fileNames]
        #self.fileNames = [file for file in self.fileNames if "missing" not in file]
        # Number of classes
        self.numClasses = numClasses
        self.swin = swin

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        # Create dict to ascribe labels
        # Creating a new dictionary with the same keys but with classification vectors as values
        nameDict = {"NORM": [1,0,0,0,0], 
                    "MI":[0,1,0,0,0], 
                    "STTC":[0,0,1,0,0],
                    "CD":[0,0,0,1,0], 
                    "HYP":[0,0,0,0,1]}
        # Get class label
        fileKey = self.fileNames[idx].split("/")[-1].split("-")[-2]
        label = nameDict[fileKey]
        # Load ECG data
        ecgData = np.load(self.fileNames[idx])
        # Expand dims to make it 3D
        # Reduce data shape (squeeze)
        # Transpose axis to proper format
        # and make it float32, not double
        ecgData = ecgData.transpose(1, 0).astype(np.float32)
        ecgLabels = np.array(label, dtype=np.float32)
        # Make them torch tensors
        ecgData = torch.from_numpy(ecgData)
        ecgLabels = torch.from_numpy(ecgLabels)
        if self.swin:
            ecgData = ecgData.unsqueeze(0)
            ecgData = torch.cat([ecgData, ecgData, ecgData], dim=0)
        return ecgData, ecgLabels
    
##############
### MODELS ###
##############

################
### CNNAttia ###
################
class CNN2023Attia(torch.nn.Module):
    def __init__(self, numClasses):
        super().__init__()
        self.convMaxBlock1 = ConvMaxBlock(inChannels=12, numFilters=16, kernelSize=5, maxPoolKernel=2, targetLength=1250, stride=5)
        self.convMaxBlock2 = ConvMaxBlock(inChannels=16, numFilters=16, kernelSize=5, maxPoolKernel=2, targetLength=625)
        self.convMaxBlock3 = ConvMaxBlock(inChannels=16, numFilters=32, kernelSize=5, maxPoolKernel=4, targetLength=312)
        self.convMaxBlock4 = ConvMaxBlock(inChannels=32, numFilters=32, kernelSize=3, maxPoolKernel=2, targetLength=156)
        self.convMaxBlock5 = ConvMaxBlock(inChannels=32, numFilters=64, kernelSize=3, maxPoolKernel=2, targetLength=78)
        self.convMaxBlock6 = ConvMaxBlock(inChannels=64, numFilters=64, kernelSize=3, maxPoolKernel=4, targetLength=39)

        self.convBlock1 = ConvBlockAlone(inChannels=64, numFilters=128, kernelSize=3, targetLength=19)
        
        self.ffn1 = FFN(inFeatures=128*19, outFeatures=256)
        self.ffn2 = FFN(inFeatures=256, outFeatures=128)
        self.out = nn.Linear(in_features=128, out_features=numClasses)

    def forward(self, x):
        x = self.convMaxBlock1(x)
        x = self.convMaxBlock2(x)
        x = self.convMaxBlock3(x)
        x = self.convMaxBlock4(x)
        x = self.convMaxBlock5(x)
        x = self.convMaxBlock6(x)
        x = self.convBlock1(x)
        x = x.view(x.size(0), -1)
        x = self.ffn1(x)
        x = self.ffn2(x)
        x = self.out(x)
        return x


class ConvMaxBlock(torch.nn.Module):
    def __init__(self, inChannels, numFilters, kernelSize, maxPoolKernel, targetLength, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=inChannels, out_channels=numFilters, 
                               kernel_size=kernelSize, stride=stride)
        self.bn = nn.BatchNorm1d(num_features=numFilters)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=maxPoolKernel)
        self.targetLength = targetLength
        self.maxPoolKernel = maxPoolKernel

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = padSeqSymm(x, self.targetLength, 2)
        x = self.relu(x)
        x = self.maxPool(x)
        x = padSeqSymm(x, self.targetLength // self.maxPoolKernel, 2)
        return x
    

class ConvBlockAlone(torch.nn.Module):
    def __init__(self, inChannels, numFilters, kernelSize, targetLength, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=inChannels, out_channels=numFilters, 
                               kernel_size=kernelSize, stride=stride)
        self.bn = nn.BatchNorm1d(num_features=numFilters)
        self.relu = nn.ReLU()
        self.targetLength = targetLength

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = padSeqSymm(x, self.targetLength, 2)
        x = self.relu(x)
        return x


class FFN(torch.nn.Module):
    def __init__(self, inFeatures, outFeatures):
        super().__init__()
        self.lin = nn.Linear(inFeatures, outFeatures)
        self.bn = nn.BatchNorm1d(num_features=outFeatures)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.lin(x)))
        return x

#########################
### Gated Transformer ###
#########################
def position_encode(x):
    pe = torch.ones_like(x[0])
    position = torch.arange(0, x.shape[1]).unsqueeze(-1)
    temp = torch.Tensor(range(0, x.shape[-1], 2))
    temp = temp * -(math.log(10000) / x.shape[-1])
    temp = torch.exp(temp).unsqueeze(0)
    temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
    pe[:, 0::2] = torch.sin(temp)
    pe[:, 1::2] = torch.cos(temp)
    return x + pe


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dimModel, q, v, h, stage):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.q = q
        self.W_Q = torch.nn.Linear(in_features=dimModel, out_features=q * h)
        self.W_K = torch.nn.Linear(in_features=dimModel, out_features=q * h)
        self.W_V = torch.nn.Linear(in_features=dimModel, out_features=v * h)
        self.W_out = torch.nn.Linear(in_features=v * h, out_features=dimModel)
        self.inf = -2**32+1
        self.stage = stage
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        Q = torch.cat(self.W_Q(x).chunk(self.h, dim=-1), dim=0)
        K = torch.cat(self.W_K(x).chunk(self.h, dim=-1), dim=0)
        V = torch.cat(self.W_V(x).chunk(self.h, dim=-1), dim=0)
        score = torch.matmul(Q, K.transpose(-1, -2))  # / torch.sqrt(torch.Tensor(self.q)).to(self.device)
        heatmapScore = score
        if self.stage == 'train':
            mask = torch.ones_like(score[0])
            mask = mask.tril(diagonal=0)
            score = torch.where(mask > 0, score, (torch.ones_like(mask) * self.inf).to(self.device))
        score = torch.nn.functional.softmax(score, dim=-1)
        weight_V = torch.cat(torch.matmul(score, V).chunk(self.h, dim=0), dim=-1)
        out = self.W_out(weight_V)
        return out, heatmapScore


class PositionFeedforward(torch.nn.Module):
    def __init__(self, dimModel, dimHidden=2048):
        super(PositionFeedforward, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=dimModel, out_features=dimHidden)
        self.linear2 = torch.nn.Linear(in_features=dimHidden, out_features=dimModel)
        self.relu = torch.nn.ReLU()
        self.layernorm = torch.nn.LayerNorm(normalized_shape=dimModel)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.layernorm(x + residual)
        return x


class Gated2TowerTransformer(torch.nn.Module):
    def __init__(self, dimModel, dimHidden, dimFeature, dimTimestep, q, v, h, N, 
                 classNum, stage, dropout=0.2):
        super(Gated2TowerTransformer, self).__init__()

        self.stage = stage
        self.timestepEmbed = Embedding(dimFeature=dimFeature, dimTimestep=dimTimestep, 
                                       dimModel=dimModel, wise='timestep')
        self.featureEmbed = Embedding(dimFeature=dimFeature, dimTimestep=dimTimestep, 
                                      dimModel=dimModel, wise='feature')

        self.timestepEncoderList = torch.nn.ModuleList([Encoder(
            dimModel=dimModel,
            dimHidden=dimHidden,
            q=q,
            v=v,
            h=h,
            dropout=dropout,
            stage=stage) for _ in range(N)])

        self.featureEncoderList = torch.nn.ModuleList([Encoder(
            dimModel=dimModel,
            dimHidden=dimHidden,
            q=q,
            v=v,
            h=h,
            dropout=dropout,
            stage=stage) for _ in range(N)])

        self.gate = torch.nn.Linear(in_features=dimTimestep * dimModel + dimFeature * dimModel, out_features=2)
        self.linear_out = torch.nn.Linear(in_features=dimTimestep * dimModel + dimFeature * dimModel,
                                          out_features=classNum)

    def forward(self, x):
        xTimestep, _ = self.timestepEmbed(x)
        xFeature, _ = self.featureEmbed(x)
        for encoder in self.timestepEncoderList:
            xTimestep, heatmap = encoder(xTimestep)
        for encoder in self.featureEncoderList:
            xFeature, heatmap = encoder(xFeature)
        xTimestep = xTimestep.reshape(xTimestep.shape[0], -1)
        xFeature = xFeature.reshape(xFeature.shape[0], -1)
        gate = torch.nn.functional.softmax(self.gate(torch.cat([xTimestep, xFeature], dim=-1)), dim=-1)
        gateOut = torch.cat([xTimestep * gate[:, 0:1], xFeature * gate[:, 1:2]], dim=-1)
        out = self.linear_out(gateOut)
        return out


class Embedding(nn.Module):   
    def __init__(self, dimFeature, dimTimestep, dimModel, wise):
        super(Embedding, self).__init__()
        assert wise in ['timestep', 'feature'], 'Embedding wise error!'
        self.wise = wise
        if wise == 'timestep':
            self.embedding = nn.Linear(dimFeature, dimModel)
        elif wise == 'feature':
            self.embedding = nn.Linear(dimTimestep, dimModel)

    def forward(self, x):
        if self.wise == 'feature':
            x = self.embedding(x)
        elif self.wise == 'timestep':
            x = x.transpose(-1, -2)  # Swap the last two dimensions
            x = self.embedding(x)
            x = position_encode(x)
        return x, None


class Encoder(torch.nn.Module):
    def __init__(self, q, v, h, dimModel, dimHidden, stage, dropout=0.2):
        super(Encoder, self).__init__() 
        self.MHA = MultiHeadAttention(dimModel=dimModel, q=q, v=v, h=h, stage=stage)
        self.feedforward = PositionFeedforward(dimModel=dimModel, dimHidden=dimHidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layernorm = torch.nn.LayerNorm(dimModel)
        self.stage = stage

    def forward(self, x):
        residual = x
        x, heatmapScore = self.MHA(x)
        x = self.dropout(x)
        x = self.layernorm(x + residual)
        x = self.feedforward(x)
        return x, heatmapScore

##################
### MLSTMFCNN ###
##################

class LSTMConvBlock(nn.Module):
    def __init__(self, ni, no, ks):
        super(LSTMConvBlock, self).__init__() 
        self.conv = nn.Conv1d(ni, no, ks, padding='same')
        self.bn = nn.BatchNorm1d(no, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SqueezeExciteBlock(nn.Module):
    def __init__(self, ni, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = GAP1d(1)
        self.fc = nn.Sequential(nn.Linear(ni, ni // reduction, bias=False), nn.ReLU(),  nn.Linear(ni // reduction, ni, bias=False), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y).unsqueeze(2)
        return x * y.expand_as(x)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__() 
        self.dim = dim
    def forward(self, x): 
        return torch.cat(x, dim=self.dim)


class Reshape(nn.Module):
    def __init__(self, *shape): 
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(x.shape[0], -1) if not self.shape else x.reshape(-1) if self.shape == (-1,) else x.reshape(x.shape[0], *self.shape)


class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__() 
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()
    def forward(self, x):
        return self.flatten(self.gap(x))


class MLSTMFCN(nn.Module):
    def __init__(self, numClasses):
        super(MLSTMFCN, self).__init__()
        # LSTM
        self.LSTM = nn.LSTM(input_size=12, hidden_size=64, num_layers=1, batch_first=True)
        self.LSTMdropout = nn.Dropout(0.8)

        # FCN
        self.convblock1 = LSTMConvBlock(12, 128, 7)
        self.se1 = SqueezeExciteBlock(128, 16)
        self.convblock2 = LSTMConvBlock(128, 256, 5)
        self.se2 = SqueezeExciteBlock(256, 16)
        self.convblock3 = LSTMConvBlock(256, 128, 3)
        self.gap = GAP1d(1)

        # Common
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(100 + 128, numClasses)

    def forward(self, x):
        # RNN
        LSTMInput = torch.permute(x, (0, 2, 1))  # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, (hn, cn) = self.LSTM(LSTMInput)
        y = self.LSTMdropout(output[:, -1, :])  # Using the output at the last time step

        # FCN
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x).squeeze(-1)  # Remove the last dimension after GAP

        # Concat
        combined = self.concat((y, x))
        combined = self.fc_dropout(combined)
        combined = self.fc(combined)
        return combined


#########################
### Original CNN 2020 ###
#########################

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, targetLength):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=inChannels, out_channels=outChannels, 
                               kernel_size=15, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=outChannels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=outChannels, out_channels=outChannels, 
                               kernel_size=15, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(num_features=outChannels)
        self.relu2 = nn.ReLU()
        self.convRes = nn.Conv1d(in_channels=inChannels, out_channels=outChannels, 
                                 kernel_size=15, stride=2)
        self.bnRes = nn.BatchNorm1d(num_features=outChannels)
        self.relu3 = nn.ReLU()
        self.targetLength = targetLength

    def forward(self, x):
        y = x.clone()
        x = self.bn1(self.conv1(x))
        x = padSeqSymm(x, self.targetLength, 2)
        x = self.relu1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        y = self.bnRes(self.convRes(y))
        y = padSeqSymm(y, self.targetLength, 2)
        x = y + x
        x = self.relu3(x)
        return x


class IDENBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=inChannels, out_channels=outChannels, 
                               kernel_size=15, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(num_features=outChannels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = x.clone()
        x = self.relu1(self.bn1(self.conv1(x)))
        x = y + x
        x = self.relu2(x)
        return x


class CNNBlock(nn.Module):
    def __init__(self, inChannels, outChannels, targetLength):
        super().__init__()
        self.convBlock = ConvBlock(inChannels, outChannels, targetLength) 
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        if inChannels == outChannels:
            self.IDEN = IDENBlock(inChannels, outChannels)
        else:
            self.IDEN = IDENBlock(inChannels*2, outChannels)
        self.targetLength = targetLength

    def forward(self, x):
        x = self.convBlock(x)
        x = padSeqSymm(x, self.targetLength, 2)
        x = self.pool(x)
        x = padSeqSymm(x, self.targetLength//2, 2)
        x = self.IDEN(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.lin = nn.Linear(inChannels, outChannels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.6)

    def forward(self, x):
        x = self.drop(self.relu(self.lin(x)))
        return x


class ECGCNNClassifier(nn.Module):
    '''Model from paper:
    Automatic multilabel electrocardiogram diagnosis of heart rhythm or conduction abnormalities with deep learning: a cohort study
    It is in the supplementary material'''
    def __init__(self, numClasses):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=15, stride=1, padding='same')
        self.bn = nn.BatchNorm1d(num_features=64)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.convBlock1 = CNNBlock(inChannels=64, outChannels=64, targetLength=625)
        self.convBlock2 = CNNBlock(inChannels=64, outChannels=128, targetLength=156)
        self.convBlock3 = CNNBlock(inChannels=128, outChannels=256, targetLength=39)
        self.convBlock4 = CNNBlock(inChannels=256, outChannels=512, targetLength=9)
        self.avgPool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.Dense1 = DenseBlock(inChannels=1024, outChannels=512)
        self.Dense2 = DenseBlock(inChannels=512, outChannels=512)
        self.fc = nn.Linear(512, numClasses)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.maxPool(self.relu(x))
        x = padSeqSymm(x, 1250, 2)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.avgPool(x)
        x = self.flatten(x)
        x = self.Dense1(x)
        x = self.Dense2(x)
        x = self.fc(x)
        return x
    
#############################################

class ECGSimpleClassifier(nn.Module):
    '''Random Model I came up with to see that
        my data runs'''
    def __init__(self, numClasses):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, stride=5)
        init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=10)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=5)
        init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.fc1 = nn.Linear(128, 64)
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc = nn.Linear(64, numClasses)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc(x)
        return x