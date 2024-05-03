import plotly.graph_objects as go
import numpy as np
import os
from plotly.subplots import make_subplots
import torch
import os
from models import padSeqSymm
import shutil
# Load your data
data1 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold10/missingLead3-AVNRT-patient331.npy")[:, 0]
data2 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold6/missingLead4-denoised-AVNRT-patient137.npy")[:, 0]
data3 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold7/missingLead4-normal-patient1036.npy")[:, 0]
data4 = np.load("/home/tzikos/Desktop/Data/Berts final/pre/fold3/missingLead12-concealed-patient341.npy")[:, 0]
data5 = np.load("/home/tzikos/Desktop/Data/Berts final/pre/fold8/missingLead2-normal-patient9031.npy")[:, 0]
data6 = np.load("/home/tzikos/Desktop/Data/Berts final/pre/fold9/missingLead5-normal-patient6250.npy")[:, 0]
data7 = np.load("/home/tzikos/Desktop/Data/PTBXL Diagnostic torch/val/orig-CD-235.npy")[:, 0]
data8 = np.load("/home/tzikos/Desktop/Data/PTBXL Diagnostic torch/train/orig-STTC-167.npy")[:, 0]
data9 = np.load("/home/tzikos/Desktop/Data/PTBXL Diagnostic torch/train/orig-HYP-313.npy")[:, 0]

# Create subplots
fig = make_subplots(rows=3, cols=3, shared_xaxes=True)

# Add traces, specifying which subplot they belong to
fig.add_trace(go.Scatter(y=data1, name=f'Mean: {np.mean(data1)}, Sigma: {np.std(data1)}'), row=1, col=1)
fig.add_trace(go.Scatter(y=data2, name=f'Mean: {np.mean(data2)}, Sigma: {np.std(data2)}'), row=2, col=1)
fig.add_trace(go.Scatter(y=data3, name=f'Mean: {np.mean(data3)}, Sigma: {np.std(data3)}'), row=3, col=1)
fig.add_trace(go.Scatter(y=data4, name=f'Mean: {np.mean(data4)}, Sigma: {np.std(data4)}'), row=1, col=2)
fig.add_trace(go.Scatter(y=data5, name=f'Mean: {np.mean(data5)}, Sigma: {np.std(data5)}'), row=2, col=2)
fig.add_trace(go.Scatter(y=data6, name=f'Mean: {np.mean(data6)}, Sigma: {np.std(data6)}'), row=3, col=2)
fig.add_trace(go.Scatter(y=data7, name=f'Mean: {np.mean(data7)}, Sigma: {np.std(data7)}'), row=1, col=3)
fig.add_trace(go.Scatter(y=data8, name=f'Mean: {np.mean(data8)}, Sigma: {np.std(data8)}'), row=2, col=3)
fig.add_trace(go.Scatter(y=data9, name=f'Mean: {np.mean(data9)}, Sigma: {np.std(data9)}'), row=3, col=3)

# Update layout for shared x-axis and set titles
fig.update_layout(title_text=f"Col 1 tachy - Col 2 Pre - Col 3 PTBXL")
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_yaxes(title_text="Amplitude")

# Adjust subplot spacing and legend
fig.update_layout(showlegend=True)
fig.update_traces(mode='lines')

# Save the figure as an HTML file
os.makedirs("/home/tzikos/UsefulFigs", exist_ok=True)
fig.write_html(f"/home/tzikos/UsefulFigs/BertsExample.html")

