import torch
from captum.attr import IntegratedGradients, Saliency, DeepLift, GradientShap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import ECGCNNClassifier
import numpy as np
import torch.nn.functional as F
import os 
from utility import seedEverything
import captum
from captum.attr import visualization as viz
import matplotlib.pyplot as plt

# Seed everything for os parsing of same files
seedEverything(44)




# Path of fold data wanted
PATH = "/home/tzikos/Desktop/Data/Berts final/pre/fold5"

# Number in Batch to take
NUM_CLASSES = 2
DICT = {0:"AVNRT", 1:"AVRT"}
# Window size for averaging filter
WINDOW_SIZE = 51
# Saving the attributions
SAVE_OPT = True
# Visualization factor
VIS_FACTOR = 10
# Load model and weights
model = ECGCNNClassifier(numClasses=NUM_CLASSES)
model.load_state_dict(torch.load('/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold4_tachy_B64_L1e-06_08-07-24-09-15.pth'))
# Load Integrated Gradients object from Captum
#ig = IntegratedGradients(model)
#ig = Saliency(model)
#ig = DeepLift(model)
ig = GradientShap(model)


# Initialize lists to save 
# class files
allClassFiles = []

# Save Lead 2 for each class
smoothed_2 = None

# Loop through the classes
for i in range(NUM_CLASSES):
    # Get all files
    allFiles = os.listdir(PATH)
    allFiles = [os.path.join(PATH, file) for file in allFiles]
    # Filter out missing leads and irrelevant classes
    classFiles = [file for file in allFiles if ("missing" not in file) and (DICT[i] in file)]
    allClassFiles.append(classFiles)
class_signals = np.zeros((NUM_CLASSES, 12, 5000))
# Loop through classes
for i in range(NUM_CLASSES):
    # Take only the first BATCH files
    class_signals[i, :, :] = np.load(allClassFiles[i][0]).T 
for i in range(NUM_CLASSES):
    # Ascribe the correct class for Captum
    target_class = i
    # Initialize lists to hold the attributions
    lead_attributions = []
    for lead in range(12):
        # Initialize tensor to hold the lead data
        print(f"Processing class {DICT[i]}, lead {lead+1}")
        # Convert numpy array to tensor and make it float
        lead_signal_tensor = torch.from_numpy(class_signals[i]).unsqueeze(0).float()
        #print(lead_signal_tensor.size())
        # Get the attributions
        attribution = ig.attribute(lead_signal_tensor, target=target_class, 
                                   baselines=torch.zeros(1,12,5000).float(), 
                                   n_samples=50,
                                   stdevs=0.1*torch.std(lead_signal_tensor).item()
                                   #show_progress=True,
                                   )
        print("no error in attributions")
        lead_attributions.append(attribution)
        # Initialize plotly figure to write per lead, per class results
        # Create averaging filter
        avg_filter = torch.ones(1, WINDOW_SIZE, dtype=torch.float32) / WINDOW_SIZE
        #print(avg_filter.size())
        # Extend attributions
        #print(lead_attributions[0].size())
        ecg_attrib = lead_attributions[0][0, lead, :].unsqueeze(0).detach().numpy()
        #print(ecg_attrib.shape)
        # Get lead data for the first ECG in the batch
        class_signal = class_signals[i, lead, :]
        #print(class_signal.shape)
        # Normalize per ECG amplitude at each timestep
        ecg_temp = ecg_attrib * VIS_FACTOR
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
        print(smoothed.size())
        if lead == 1:
            smoothed_2 = smoothed
    viz.visualize_timeseries_attr(
        smoothed_2[1200:3800].unsqueeze(1), 
        torch.from_numpy(class_signals[i, 1, 1200:3800].T).unsqueeze(1), 
        fig_size=(20, 20),
        title = f"Lead 2 of Class {DICT[i]}",
    )
