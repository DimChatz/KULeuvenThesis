import torch
from captum.attr import IntegratedGradients
import plotly.graph_objects as go
from models import ECGCNNClassifier
import numpy as np
import torch.nn.functional as F
import os 

PATH = "/home/tzikos/Desktop/Data/Berts final/pre/fold5"
BATCH = 32
NUM_CLASSES = 2
DICT = {0:"AVNRT", 1:"AVRT"}
WINDOW_SIZE = 101
model = ECGCNNClassifier(numClasses=NUM_CLASSES)
model.load_state_dict(torch.load('/home/tzikos/Desktop/weights/Models/ECGCNNClassifier_fold5_pre_B64_L1e-05_01-07-24-09-27.pth'))
ig = IntegratedGradients(model)

allClassFiles = []
for i in range(NUM_CLASSES):
    allFiles = os.listdir(PATH)
    allFiles = [os.path.join(PATH, file) for file in allFiles]
    classFiles = [file for file in allFiles if ("missing" not in file) and (DICT[i] in file)]
    allClassFiles.append(classFiles)

class_signals = np.zeros((NUM_CLASSES, BATCH, 12, 5000))
for i in range(NUM_CLASSES):
    for j in range(BATCH):
        for file in allClassFiles[i]:
            class_signals[i, j, :, :] = np.load(file).T
    target_class = i
    lead_attributions = []
    lead_attr_sums = []
    for lead in range(12):
        lead_signal = np.zeros((BATCH, 12, 5000))
        lead_signal[:, lead, :] = class_signals[i, :, lead, :]
        attribution = ig.attribute(torch.from_numpy(lead_signal), target=target_class)
        lead_attributions.append(attribution)
        lead_attr_sums.append(attribution.sum(dim=2).sum(dim=1))

    print(lead_attr_sums)

    for lead in range(12):
        fig = go.Figure()
        avg_filter = torch.ones(WINDOW_SIZE, dtype=torch.float32) / WINDOW_SIZE
        ecg_temp = lead_attributions[lead][:, lead].detach().numpy() / np.max(np.abs(lead_attributions[lead][:, lead].detach().numpy())) / (class_signals[i, :, lead].numpy() / np.max(np.abs(class_signals[i, :, lead].numpy())))
        ecg_temp = torch.from_numpy(ecg_temp).view(1, 1, -1).float()
        smoothed = F.conv1d(ecg_temp, avg_filter.view(1, 1, -1), padding=WINDOW_SIZE//2)
        smoothed = smoothed.view(-1)
        padded_smoothed = F.pad(smoothed, (WINDOW_SIZE//2, WINDOW_SIZE//2), mode='constant', value=0)
        padded_smoothed[:WINDOW_SIZE//2] = smoothed[0].item()
        padded_smoothed[-WINDOW_SIZE//2:] = smoothed[-1].item()
        fig.add_trace(go.Scatter(y=padded_smoothed.numpy(), name=f"Lead {lead+1} Attributions"))
        fig.add_trace(go.Scatter(y=class_signals[i, :, lead, :].numpy() / np.max(np.abs(class_signals[i, 0, lead, :].numpy())), name=f"Lead {lead+1} ECG Signal"))    
        fig.update_layout(title=f"Lead {lead+1} ECG Signal and Attributions")
        #fig.show()

    lead_attr_sums = [x.numpy() for x in lead_attr_sums]
    fig = go.Figure(data=[go.Bar(x=list(range(1,len(lead_attr_sums)+1)), y=lead_attr_sums)])
    fig.update_layout(
        title='Bar Chart of ECG Lead Activation Sums',
        xaxis_title='Lead Number',
        yaxis_title='Activation Sum',
        xaxis=dict(
            tickmode='linear',
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
    fig.show()
    fig.write_html("/home/tzikos/xplainability/histogram_binary_AVNRT_AVRT_pre.html")
