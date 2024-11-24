
import torch
from captum.attr import IntegratedGradients, Saliency, DeepLift, GradientShap
import plotly.graph_objects as go
from models import ECGCNNClassifier
import numpy as np
import torch.nn.functional as F
import os
from utility import seedEverything
import captum
from itertools import chain


# Seed everything for os parsing of same files
seedEverything(42)


def fileAccumulator(path, string, numFold):
    trainFilesList = []
    for folder in os.listdir(path):
        #if str(f"fold{numFold}") in folder or str(f"fold{numFold+1}") in folder:
            #continue
        trainPath = os.path.join(path, folder)
        trainFiles = os.listdir(trainPath)
        trainFiles = [os.path.join(trainPath, file) for file in trainFiles]
        trainFiles = [file for file in trainFiles if "missing" not in file]
        trainFiles = [file for file in trainFiles if string in file]
        trainFilesList.append(trainFiles)
    trainFilesList = list(chain.from_iterable(trainFilesList))
    return trainFilesList


# Globals - to be changed per case
# Path of fold to consider
PATH = "/home/tzikos/Desktop/Data/Berts final/pre/"
# Batch size to take - handle per model for OOM issues
BATCH = 128
NUM_CLASSES = 2
DICT = {0:"AVNRT", 1:"AVRT"}
# Show class diffs (Only for Binary Data)
SHOW_DIFF = True
# Load model and weights
model = ECGCNNClassifier(numClasses=NUM_CLASSES)
model.load_state_dict(torch.load('/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold2_pre_B64_L1e-05_01-07-24-09-27.pth'))
# Load Integrated Gradients object from Captum
ig = GradientShap(model)

# Initialize lists to save 
# files that will be loaded...
allClassFiles = []
#... as well as captums differences for each class
diff_of_sums = []

# Loop through the classes
for i in range(NUM_CLASSES):
    # Get all files
    allFiles = fileAccumulator(PATH, DICT[i], 0)
    allClassFiles.append(allFiles)
# np array to hold the ECG data
class_signals = np.zeros((NUM_CLASSES, BATCH, 12, 5000))
# Initialize plotly figure to write per lead, per class results
fig = go.Figure()

# Loop through classes
for i in range(NUM_CLASSES):
    # Take only the first BATCH files
    allClassFiles[i] = allClassFiles[i][:BATCH]
    # Load the ECG data
    for file in allClassFiles[i]:
        class_signals[i, :, :, :] = np.load(file).T 
for i in range(NUM_CLASSES):
    # Define target class
    target_class = i
    # Initialize lists to hold the attribution sums
    lead_attr_sums = []
    # Loop through leads
    # Ensure tensor is tensor and float type
    lead_signal_tensor = torch.from_numpy(class_signals[i, :, :, :]).float()
    # Get the attributions and store them
    attribution = ig.attribute(lead_signal_tensor, target=target_class,
                                baselines=torch.zeros(1,12,5000).float(),
                                n_samples=6,
                                stdevs=0.1*torch.std(lead_signal_tensor).item())
    print("no error in attributions")
    #print(attribution.sum((0,2)).size())
    lead_attr_sums = attribution.sum((0,2))
    # Transform them to numpy to be visualy callable
    lead_attr_sums = [x.detach().numpy() for x in lead_attr_sums]
    # Store the sums for each lead to differentiate between classes
    diff_of_sums.append(lead_attr_sums)
    # Add bar chart for Class 1
    fig.add_trace(go.Bar(
        x=list(range(1, len(lead_attr_sums) + 1)),
        y=lead_attr_sums,
        name=f"Class {DICT[i]}"
    ))
    
if SHOW_DIFF:
    # Create differentiations
    diff_list = [diff_of_sums[0][i]-diff_of_sums[1][i] for i in range(len(diff_of_sums[0]))]
    # Add bar chart for Differences
    fig.add_trace(go.Bar(
        x=list(range(1, len(diff_list) + 1)),
        y=diff_list,
        name="Differences"
    ))

# Update layout with the provided axis settings
fig.update_layout(
    title_text="Sums and Differences of Activation Gradients for Each Class",
    barmode='group',  # Group bars side by side
    xaxis=dict(
        title='Lead',
        tick0=1,
        dtick=1,
        zeroline=True,
        showline=True,
        showgrid=True
    ),
    yaxis=dict(
        title='Sum of Attributions',
        zeroline=True,
        showline=True,
        showgrid=True
    )
)

# Show figure
fig.show()
# Save figure
fig.write_html(f"/home/tzikos/xplainability/histogram_binary_AVNRT_AVRT_pre_class_diffs.html")
