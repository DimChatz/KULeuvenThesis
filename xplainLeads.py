import torch
from captum.attr import IntegratedGradients
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import ECGCNNClassifier
import numpy as np
import torch.nn.functional as F
import os
from utility import seedEverything

seedEverything(42)

PATH = "/home/tzikos/Desktop/Data/Berts final/pre/fold5"
BATCH = 16
NUM_CLASSES = 2
DICT = {0:"AVNRT", 1:"AVRT"}
WINDOW_SIZE = 501
model = ECGCNNClassifier(numClasses=NUM_CLASSES)
model.load_state_dict(torch.load('/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold5_pre_B64_L1e-05_01-07-24-09-27.pth'))
ig = IntegratedGradients(model)

allClassFiles = []
diff_of_sums = []

for i in range(NUM_CLASSES):
    allFiles = os.listdir(PATH)
    allFiles = [os.path.join(PATH, file) for file in allFiles]
    classFiles = [file for file in allFiles if ("missing" not in file) and (DICT[i] in file)]
    allClassFiles.append(classFiles)
#print(len(allClassFiles))
class_signals = np.zeros((NUM_CLASSES, BATCH, 12, 5000))
fig = go.Figure()
for i in range(NUM_CLASSES):
    allClassFiles[i] = allClassFiles[i][:BATCH]
    for file in allClassFiles[i]:
        class_signals[i, :, :, :] = np.load(file).T 
    #print(len(allClassFiles[i]))
    target_class = i
    lead_attributions = []
    lead_attr_sums = []
    # Create a single subplot
    for lead in range(12):
        lead_signal = np.zeros((BATCH, 12, 5000))
        lead_signal[:, lead, :] = class_signals[i, :, lead, :]
        print(f"Processing class {i+1}, lead {lead+1}")
        try:
            lead_signal_tensor = torch.from_numpy(lead_signal).float()  # Ensure tensor is float type
            attribution = ig.attribute(lead_signal_tensor, target=target_class)
            print("no error in attributions")
            lead_attributions.append(attribution)
            lead_attr_sums.append(attribution[:,lead,:].sum())
        except Exception as e:
            print(f"Error in attributions for class {i}, lead {lead}: {e}")
    lead_attr_sums = [x.numpy() for x in lead_attr_sums]
    diff_of_sums.append(lead_attr_sums)
    # Add bar chart for Class 1
    fig.add_trace(go.Bar(
        x=list(range(1, len(lead_attr_sums) + 1)),
        y=lead_attr_sums,
        name=f"Class {DICT[i]}"
    ))
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
fig.write_html(f"/home/tzikos/xplainability/histogram_binary_AVNRT_AVRT_pre_class_diffs.html")