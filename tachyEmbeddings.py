import plotly.graph_objects as go
import numpy as np
import os
from plotly.subplots import make_subplots
import torch
from models import padSeqSymm
'''
# Load your data
data1 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold1/normalized-AVNRT-patient5.npy")[:, 1]
data2 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold1/normalized-denoised-AVNRT-patient5.npy")[:, 1]
data3 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold1/missingLead1-AVNRT-patient5.npy")[:, 1]
data4 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold1/missingLead1-denoised-AVNRT-patient5.npy")[:, 1]
data5 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold1/augmented0.12-AVNRT-patient5.npy")[:, 1]
data6 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold1/augmented0.12-denoised-AVNRT-patient5.npy")[:, 1]
data7 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold1/augmented0.12-missingLead1-AVNRT-patient5.npy")[:, 1]
data8 = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold1/augmented0.12-missingLead1-denoised-AVNRT-patient5.npy")[:, 1]

print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)
print(data5.shape)
print(data6.shape)
print(data7.shape)
print(data8.shape)

# Create subplots
fig = make_subplots(rows=4, cols=2, shared_xaxes=True)

# Add traces, specifying which subplot they belong to
fig.add_trace(go.Scatter(y=data1, name='Normalized Patient Lead 2'), row=1, col=1)
fig.add_trace(go.Scatter(y=data2, name='Denoised Normalized Lead 2'), row=1, col=2)
fig.add_trace(go.Scatter(y=data3, name='Missing Lead 2'), row=2, col=1)
fig.add_trace(go.Scatter(y=data3, name='Denoised Missing Lead 2'), row=2, col=2)
fig.add_trace(go.Scatter(y=data5, name='Augmented Lead 2'), row=3, col=1)
fig.add_trace(go.Scatter(y=data6, name='Denoised Lead 2'), row=3, col=2)
fig.add_trace(go.Scatter(y=data7, name='Augmented Missing Lead 2'), row=4, col=1)
fig.add_trace(go.Scatter(y=data8, name='Denoised Augmented Missing Lead 2'), row=4, col=2)

# Update layout for shared x-axis and set titles
fig.update_layout(title_text="Patient Data Pipeline of Lead 2 in case of missing lead 1")
fig.update_xaxes(title_text="Time", row=4, col=1)
fig.update_yaxes(title_text="Amplitude")

# Adjust subplot spacing and legend
fig.update_layout(showlegend=True)
fig.update_traces(mode='lines')

# Save the figure as an HTML file
os.makedirs("/home/tzikos/UsefulFigs", exist_ok=True)
fig.write_html("/home/tzikos/UsefulFigs/Lead2PreprocWithDenoised.html")
'''

example =  torch.from_numpy(np.ones((1, 3, 3)))
example_padded = padSeqSymm(batch=example, targetLength=5, dimension=2)
print(example)
print(example_padded)