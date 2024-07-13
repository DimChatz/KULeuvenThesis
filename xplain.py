import torch
from captum.attr import IntegratedGradients
import plotly.graph_objects as go
from models import ECGCNNClassifier
import numpy as np
import torch.nn.functional as F
import os 

PATH = "/home/tzikos/Desktop/Data/Berts final/pre/fold5"
BATCH = 16
NUM_CLASSES = 2
DICT = {0:"AVNRT", 1:"AVRT"}
WINDOW_SIZE = 101
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
for i in range(NUM_CLASSES):
    allClassFiles[i] = allClassFiles[i][:BATCH]
    for file in allClassFiles[i]:
        class_signals[i, :, :, :] = np.load(file).T
    #print(len(allClassFiles[i]))
    target_class = i
    lead_attributions = []
    lead_attr_sums = []
    for lead in range(12):
        lead_signal = np.zeros((BATCH, 12, 5000))
        lead_signal[:, lead, :] = class_signals[i, :, lead, :]
        print(f"Processing class {i}, lead {lead}")
        try:
            lead_signal_tensor = torch.from_numpy(lead_signal).float()  # Ensure tensor is float type
            attribution = ig.attribute(lead_signal_tensor, target=target_class)
            print(attribution.size())
            print("no error in attributions")
            lead_attributions.append(attribution)
            lead_attr_sums.append(attribution[:,lead,:].sum())
        except Exception as e:
            print(f"Error in attributions for class {i}, lead {lead}: {e}")

        fig = go.Figure()
        avg_filter = torch.ones(1, WINDOW_SIZE, dtype=torch.float32) / WINDOW_SIZE
        avg_filter = avg_filter
        print(avg_filter.size())
        # Normalizing lead_attributions and class_signals
        ecg_attrib = lead_attributions[lead][0,lead,:].unsqueeze(0).detach().numpy()
        ecg_attrib_normalized = ecg_attrib / np.max(np.abs(ecg_attrib))
        print(ecg_attrib.shape)
        class_signal = class_signals[i, :, lead, :]
        class_signal_normalized = class_signal / np.max(np.abs(class_signal))
        
        ecg_temp = ecg_attrib_normalized / class_signal_normalized
        ecg_temp = torch.from_numpy(ecg_temp).view(1, 1, -1).float()
        
        smoothed = F.conv1d(torch.from_numpy(ecg_attrib).float(), avg_filter.view(1, 1, -1), padding=WINDOW_SIZE//2)
        print(smoothed.size())
        smoothed = smoothed.view(-1)
        print(smoothed.size())
        smoothed[:WINDOW_SIZE//2] = smoothed[0].item()
        smoothed[-WINDOW_SIZE//2:] = smoothed[-1].item()
        print(smoothed.size())

        fig.add_trace(go.Scatter(y=class_signal_normalized, name=f"Lead {lead+1} ECG Signal"))    
        fig.show()
        fig.add_trace(go.Scatter(y=smoothed.numpy()/np.max(smoothed.numpy()), name=f"Lead {lead+1} Attributions"))
        fig.update_layout(title=f"Lead {lead+1} ECG Signal and Attributions")
        fig.show()

    lead_attr_sums = [x.numpy() for x in lead_attr_sums]
    diff_of_sums.append(lead_attr_sums)
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
    fig.write_html(f"/home/tzikos/xplainability/histogram_binary_AVNRT_AVRT_pre_class{DICT[i]}.html")

diff_list = [diff_of_sums[0][i]-diff_of_sums[1][i] for i in range(len(diff_of_sums[0]))]
fig = go.Figure(data=[go.Bar(x=list(range(1,len(lead_attr_sums)+1)), y=diff_list)]) 
fig.update_layout(
    title='Bar Chart of ECG Lead Activation Sums Differences',
    xaxis_title='Lead Number',
    yaxis_title='Activation Sum Difference',
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
fig.write_html(f"/home/tzikos/xplainability/histogram_binary_AVNRT_AVRT_pre_class_diffs.html")