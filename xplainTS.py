import torch
from captum.attr import IntegratedGradients
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import ECGCNNClassifier
import numpy as np
import torch.nn.functional as F
import os 
from utility import seedEverything

# Seed everything for os parsing of same files
seedEverything(42)

# Path of fold data wanted
PATH = "/home/tzikos/Desktop/Data/Berts final/pre/fold5"
# Batch size - change per model to avoid OOM issues
BATCH = 20
# Number in Batch to take
ELEMENT = 0
NUM_CLASSES = 2
DICT = {0:"AVNRT", 1:"AVRT"}
# Window size for averaging filter
WINDOW_SIZE = 501
# Load model and weights
model = ECGCNNClassifier(numClasses=NUM_CLASSES)
model.load_state_dict(torch.load('/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold5_pre_B64_L1e-05_01-07-24-09-27.pth'))
# Load Integrated Gradients object from Captum
ig = IntegratedGradients(model)

# Initialize lists to save 
# class files
allClassFiles = []

# Loop through the classes
for i in range(NUM_CLASSES):
    # Get all files
    allFiles = os.listdir(PATH)
    allFiles = [os.path.join(PATH, file) for file in allFiles]
    # Filter out missing leads and irrelevant classes
    classFiles = [file for file in allFiles if ("missing" not in file) and (DICT[i] in file)]
    allClassFiles.append(classFiles)
class_signals = np.zeros((NUM_CLASSES, BATCH, 12, 5000))
# Loop through classes
for i in range(NUM_CLASSES):
    # Take only the first BATCH files
    allClassFiles[i] = allClassFiles[i][:BATCH]
    for file in allClassFiles[i]:
        class_signals[i, :, :, :] = np.load(file).T 
    # Ascribe the correct class for Captum
    target_class = i
    # Initialize lists to hold the attributions
    lead_attributions = []
    for lead in range(12):
        # Initialize tensor to hold the lead data
        lead_signal = np.zeros((BATCH, 12, 5000))
        lead_signal[:, lead, :] = class_signals[i, :, lead, :]
        print(f"Processing class {DICT[i]}, lead {lead+1}")
        try:
            # Convert numpy array to tensor and make it float
            lead_signal_tensor = torch.from_numpy(lead_signal).float()
            # Get the attributions
            attribution = ig.attribute(lead_signal_tensor, target=target_class)
            print("no error in attributions")
            lead_attributions.append(attribution)
        except Exception as e:
            print(f"Error in attributions for class {DICT[i]}, lead {lead+1}: {e}")
        # Initialize plotly figure to write per lead, per class results
        fig = go.Figure()
        # Create averaging filter
        avg_filter = torch.ones(1, WINDOW_SIZE, dtype=torch.float32) / WINDOW_SIZE
        #print(avg_filter.size())
        # Extend attributions
        ecg_attrib = lead_attributions[lead][0,lead,:].unsqueeze(0).detach().numpy()
        #print(ecg_attrib.shape)
        # Get lead data for the first ECG in the batch
        class_signal = class_signals[i, ELEMENT, lead, :]
        #print(class_signal.shape)
        # Normalize per ECG amplitude at each timestep
        ecg_temp = ecg_attrib / class_signal
        # Make it a tensor, float and add a dimension
        ecg_temp = torch.from_numpy(ecg_temp).view(1, 1, -1).float()
        # Perform the convolution
        smoothed = F.conv1d(ecg_temp, avg_filter.view(1, 1, -1), padding=WINDOW_SIZE//2)
        #print(smoothed.size())
        # Flatten the tensor
        smoothed = smoothed.view(-1)
        #print(smoothed.size())
        # Padd appropriate with the last seen value
        smoothed[:WINDOW_SIZE//2] = smoothed[0].item()
        smoothed[-WINDOW_SIZE//2:] = smoothed[-1].item()
        #print(smoothed.size())

        # Add original signal and attributions to the plot
        fig.add_trace(go.Scatter(y=class_signal, name=f"Lead {lead+1} ECG Signal"))    
        fig.add_trace(go.Scatter(y=smoothed.numpy()/np.max(smoothed.numpy()), name=f"Lead {lead+1} Attributions"))
        fig.update_layout(title=f"Lead {lead+1} ECG Signal and Attributions fo class {DICT[i]}")
        fig.show()