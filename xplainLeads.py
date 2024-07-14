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

# Globals - to be changed per case
# Path of fold to consider
PATH = "/home/tzikos/Desktop/Data/Berts final/pre/fold5"
# Batch size to take - handle per model for OOM issues
BATCH = 20
NUM_CLASSES = 2
DICT = {0:"AVNRT", 1:"AVRT"}
# Show class diffs (Only for Binary Data)
SHOW_DIFF = True
# Load model and weights
model = ECGCNNClassifier(numClasses=NUM_CLASSES)
model.load_state_dict(torch.load('/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold5_pre_B64_L1e-05_01-07-24-09-27.pth'))
# Load Integrated Gradients object from Captum
ig = IntegratedGradients(model)
# Initialize lists to save 
# files that will be loaded...
allClassFiles = []
#... as well as captums differences for each class
diff_of_sums = []

# Loop through the classes
for i in range(NUM_CLASSES):
    # Get all files
    allFiles = os.listdir(PATH)
    allFiles = [os.path.join(PATH, file) for file in allFiles]
    # Filter out missing leads and irrelevant classes    
    classFiles = [file for file in allFiles if ("missing" not in file) and (DICT[i] in file)]
    allClassFiles.append(classFiles)
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
    # Define target class
    target_class = i
    # Initialize lists to hold the attribution sums
    lead_attr_sums = []
    # Loop through leads
    for lead in range(12):
        # Initialize tensor to hold the lead data
        lead_signal = np.zeros((BATCH, 12, 5000))
        # Ascribe values
        lead_signal[:, lead, :] = class_signals[i, :, lead, :]
        print(f"Processing class {i+1}, lead {lead+1}")
        try:
            # Ensure tensor is tensor and float type
            lead_signal_tensor = torch.from_numpy(lead_signal).float()
            # Get the attributions and store them
            attribution = ig.attribute(lead_signal_tensor, target=target_class)
            print("no error in attributions")
            lead_attr_sums.append(attribution[:,lead,:].sum())
        except Exception as e:
            print(f"Error in attributions for class {i}, lead {lead}: {e}")
    # Transform them to numpy to be visualy callable
    lead_attr_sums = [x.numpy() for x in lead_attr_sums]
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
    title_text="Sums and Differences for Each Class",
    barmode='group',  # Group bars side by side
    xaxis=dict(
        tick0=1,
        dtick=1,
        zeroline=True,
        showline=True,
        showgrid=True
    ),
    yaxis=dict(
        zeroline=True,
        showline=True,
        showgrid=True
    )
)

# Show figure
fig.show()
# Save figure
fig.write_html(f"/home/tzikos/xplainability/histogram_binary_AVNRT_AVRT_pre_class_diffs.html")