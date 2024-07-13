import plotly.graph_objects as go
import numpy as np
import os
from plotly.subplots import make_subplots
from preprocessing import noiseRemover
import torch
import os
import shutil
from utility import findDuplicatePatients
#findDuplicatePatients("/home/tzikos/Desktop/Data/Berts")
meanPre = np.load(f"/home/tzikos/Desktop/weights/meanpreBerts.npy")
sigmaPre = np.load(f"/home/tzikos/Desktop/weights/sigmapreBerts.npy")
meanTachy = np.load(f"/home/tzikos/Desktop/weights/meantachyBerts.npy")
sigmaTachy = np.load(f"/home/tzikos/Desktop/weights/sigmatachyBerts.npy")
print(meanPre)
print(sigmaPre)
print(meanTachy)
print(sigmaTachy)
lead = 1
# Load your data
data1 = np.load("/home/tzikos/Desktop/Data/Berts orig/pre/orig-normal-E6C4BEFB-02C5-4F61-B9F7-2D748DE7322E_2A6M7VK6VKN2_20230828_1.npy")[:, lead]
data2 = noiseRemover(data1, lowF=0.5, highF=47, samplingF=500)
data2b = np.load("/home/tzikos/Desktop/Data/Berts final/pre/fold9/orig-normal-E6C4BEFB-02C5-4F61-B9F7-2D748DE7322E_2A6M7VK6VKN2_20230828_1.npy")[:, lead]

data3 = np.load("/home/tzikos/Desktop/Data/Berts orig/pre/orig-AVNRT-D659DA5A-12D1-4C15-A28A-6B3B7C6BA135_2AKM2SNUVHAS_19970605_1.npy")[:, lead]
data4 = noiseRemover(data3, lowF=0.5, highF=47, samplingF=500)
data4b = np.load("/home/tzikos/Desktop/Data/Berts final/pre/fold5/orig-AVNRT-D659DA5A-12D1-4C15-A28A-6B3B7C6BA135_2AKM2SNUVHAS_19970605_1.npy")[:, lead]

data5 = np.load("/home/tzikos/Desktop/Data/Berts orig/pre/orig-AVRT-434691A5-E435-4AC0-9E7E-41D3E98C3EA9_2A7C9Z5KL7F1_20210505_1.npy")[:, lead]
data6 = noiseRemover(data5, lowF=0.5, highF=47, samplingF=500)
data6b = np.load("/home/tzikos/Desktop/Data/Berts final/pre/fold9/orig-AVRT-434691A5-E435-4AC0-9E7E-41D3E98C3EA9_2A7C9Z5KL7F1_20210505_1.npy")[:, lead]

data7 = np.load("/home/tzikos/Desktop/Data/Berts orig/pre/orig-concealed-CB975F0E-39C8-42B3-8C0F-6F6DA4BEDE0B_2BJLT8NKN4KG_20080203_1.npy")[:, lead]
data8 = noiseRemover(data7, lowF=0.5, highF=47, samplingF=500)
data8b = np.load("/home/tzikos/Desktop/Data/Berts final/pre/fold7/orig-concealed-CB975F0E-39C8-42B3-8C0F-6F6DA4BEDE0B_2BJLT8NKN4KG_20080203_1.npy")[:, lead]

data9 = np.load("/home/tzikos/Desktop/Data/Berts orig/pre/orig-EAT-7996BA68-1C9C-49BF-BBC7-7105CE6B5EE0_2A1U5L5F4DFW_20070211_1.npy")[:, lead]
data10 = noiseRemover(data9, lowF=0.5, highF=47, samplingF=500)
data10b = np.load("/home/tzikos/Desktop/Data/Berts final/pre/fold5/orig-EAT-7996BA68-1C9C-49BF-BBC7-7105CE6B5EE0_2A1U5L5F4DFW_20070211_1.npy")[:, lead]

data11 = np.load("/home/tzikos/Desktop/Data/Berts orig/tachy/orig-normal-F9114E5C-32A1-4BD9-A688-A9362E171F93_2A7MWK678DML_20210127_1.npy")[:, lead]
data12 = noiseRemover(data11, lowF=0.5, highF=47, samplingF=500)
data12b = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold7/orig-normal-F9114E5C-32A1-4BD9-A688-A9362E171F93_2A7MWK678DML_20210127_1.npy")[:, lead]

data13 = np.load("/home/tzikos/Desktop/Data/Berts orig/tachy/orig-AVNRT-75E6EFE9-4EC8-4B3A-9E89-5004E6A326F2_2AX1ZWAX3DRF_20220426_2.npy")[:, lead]
data14 = noiseRemover(data13, lowF=0.5, highF=47, samplingF=500) 
data14b = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold5/orig-AVNRT-75E6EFE9-4EC8-4B3A-9E89-5004E6A326F2_2AX1ZWAX3DRF_20220426_2.npy")[:, lead]

data15 = np.load("/home/tzikos/Desktop/Data/Berts orig/tachy/orig-AVRT-677AF13F-A8AC-4948-AC70-9DA836B69456_2Q66PUJR98NH_20121125_1.npy")[:, lead]
data16 = noiseRemover(data15, lowF=0.5, highF=47, samplingF=500)
data16b = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold4/orig-AVRT-677AF13F-A8AC-4948-AC70-9DA836B69456_2Q66PUJR98NH_20121125_1.npy")[:, lead]

data17 = np.load("/home/tzikos/Desktop/Data/Berts orig/tachy/orig-concealed-778006D8-0DD3-4E6E-A0BA-8B4EC629623E_2JBK4S2QWFK2_20190325_1.npy")[:, lead]
data18 = noiseRemover(data17, lowF=0.5, highF=47, samplingF=500)
data18b = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold6/orig-concealed-778006D8-0DD3-4E6E-A0BA-8B4EC629623E_2JBK4S2QWFK2_20190325_1.npy")[:, lead]

data19 = np.load("/home/tzikos/Desktop/Data/Berts orig/tachy/orig-EAT-687F85B6-AC5D-46BF-A456-6098D50BE140_2BWM9ZH4NY3C_20070629_1.npy")[:, lead]
data20 = noiseRemover(data19, lowF=0.5, highF=47, samplingF=500)
data20b = np.load("/home/tzikos/Desktop/Data/Berts final/tachy/fold10/orig-EAT-687F85B6-AC5D-46BF-A456-6098D50BE140_2BWM9ZH4NY3C_20070629_1.npy")[:, lead]

# Create subplots
fig = make_subplots(rows=8, cols=5, shared_xaxes=True)

# Add traces, specifying which subplot they belong to
fig.add_trace(go.Scatter(y=data1, name=f'Mean: {np.mean(data1)}, Sigma: {np.std(data1)}'), row=1, col=1)
fig.add_trace(go.Scatter(y=data2, name=f'Mean: {np.mean(data2)}, Sigma: {np.std(data2)}'), row=2, col=1)
fig.add_trace(go.Scatter(y=data2b, name=f'Mean: {np.mean(data2b)}, Sigma: {np.std(data2b)}'), row=3, col=1)

fig.add_trace(go.Scatter(y=data3, name=f'Mean: {np.mean(data3)}, Sigma: {np.std(data3)}'), row=1, col=2)
fig.add_trace(go.Scatter(y=data4, name=f'Mean: {np.mean(data4)}, Sigma: {np.std(data4)}'), row=2, col=2)
fig.add_trace(go.Scatter(y=data4b, name=f'Mean: {np.mean(data4b)}, Sigma: {np.std(data4b)}'), row=3, col=2)

fig.add_trace(go.Scatter(y=data5, name=f'Mean: {np.mean(data5)}, Sigma: {np.std(data5)}'), row=1, col=3)
fig.add_trace(go.Scatter(y=data6, name=f'Mean: {np.mean(data6)}, Sigma: {np.std(data6)}'), row=2, col=3)
fig.add_trace(go.Scatter(y=data6b, name=f'Mean: {np.mean(data6b)}, Sigma: {np.std(data6b)}'), row=3, col=3)

fig.add_trace(go.Scatter(y=data7, name=f'Mean: {np.mean(data7)}, Sigma: {np.std(data7)}'), row=1, col=4)
fig.add_trace(go.Scatter(y=data8, name=f'Mean: {np.mean(data8)}, Sigma: {np.std(data8)}'), row=2, col=4)
fig.add_trace(go.Scatter(y=data8b, name=f'Mean: {np.mean(data8b)}, Sigma: {np.std(data8b)}'), row=3, col=4)

fig.add_trace(go.Scatter(y=data9, name=f'Mean: {np.mean(data9)}, Sigma: {np.std(data9)}'), row=1, col=5)
fig.add_trace(go.Scatter(y=data10, name=f'Mean: {np.mean(data10)}, Sigma: {np.std(data10)}'), row=2, col=5)
fig.add_trace(go.Scatter(y=data10b, name=f'Mean: {np.mean(data10b)}, Sigma: {np.std(data10b)}'), row=3, col=5)



fig.add_trace(go.Scatter(y=data11, name=f'Mean: {np.mean(data11)}, Sigma: {np.std(data11)}'), row=4, col=1)
fig.add_trace(go.Scatter(y=data12, name=f'Mean: {np.mean(data12)}, Sigma: {np.std(data12)}'), row=5, col=1)
fig.add_trace(go.Scatter(y=data12b, name=f'Mean: {np.mean(data12b)}, Sigma: {np.std(data12b)}'), row=6, col=1)

fig.add_trace(go.Scatter(y=data13, name=f'Mean: {np.mean(data13)}, Sigma: {np.std(data13)}'), row=4, col=2)
fig.add_trace(go.Scatter(y=data14, name=f'Mean: {np.mean(data14)}, Sigma: {np.std(data14)}'), row=5, col=2)
fig.add_trace(go.Scatter(y=data14b, name=f'Mean: {np.mean(data14b)}, Sigma: {np.std(data14b)}'), row=6, col=2)

fig.add_trace(go.Scatter(y=data15, name=f'Mean: {np.mean(data15)}, Sigma: {np.std(data15)}'), row=4, col=3)
fig.add_trace(go.Scatter(y=data16, name=f'Mean: {np.mean(data16)}, Sigma: {np.std(data16)}'), row=5, col=3)
fig.add_trace(go.Scatter(y=data16b, name=f'Mean: {np.mean(data16b)}, Sigma: {np.std(data16b)}'), row=6, col=3)

fig.add_trace(go.Scatter(y=data17, name=f'Mean: {np.mean(data17)}, Sigma: {np.std(data17)}'), row=4, col=4)
fig.add_trace(go.Scatter(y=data18, name=f'Mean: {np.mean(data18)}, Sigma: {np.std(data18)}'), row=5, col=4)
fig.add_trace(go.Scatter(y=data18b, name=f'Mean: {np.mean(data18b)}, Sigma: {np.std(data18b)}'), row=6, col=4)

fig.add_trace(go.Scatter(y=data19, name=f'Mean: {np.mean(data19)}, Sigma: {np.std(data19)}'), row=4, col=5)
fig.add_trace(go.Scatter(y=data20, name=f'Mean: {np.mean(data20)}, Sigma: {np.std(data20)}'), row=5, col=5)
fig.add_trace(go.Scatter(y=data20b, name=f'Mean: {np.mean(data20b)}, Sigma: {np.std(data20b)}'), row=6, col=5)

# Update layout for shared x-axis and set titles
fig.update_layout(title_text=f"Col 1 tachy - Col 2 Pre - Col 3 PTBXL")
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_yaxes(title_text="Amplitude")

# Adjust subplot spacing and legend
fig.update_layout(showlegend=True)
fig.update_traces(mode='lines')

# Save the figure as an HTML file
os.makedirs("/home/tzikos/UsefulFigs", exist_ok=True)
fig.write_html(f"/home/tzikos/UsefulFigs/BertsExampleDefault.html")
'''
lead=1
data1 = np.load("/home/tzikos/Desktop/Data/PTBXL Diagnostic torch/val/CD-32.npy")[:, lead]
data2 = np.load("/home/tzikos/Desktop/Data/PTBXL Diagnostic torch/val/HYP-2213.npy")[:, lead]
data3 = np.load("/home/tzikos/Desktop/Data/PTBXL Diagnostic torch/val/MI-4981.npy")[:, lead]

# Create subplots
fig = make_subplots(rows=3, cols=3, shared_xaxes=True)

# Add traces, specifying which subplot they belong to
fig.add_trace(go.Scatter(y=data1, name=f'Mean: {np.mean(data1)}, Sigma: {np.std(data1)}'), row=1, col=1)
fig.add_trace(go.Scatter(y=data2, name=f'Mean: {np.mean(data2)}, Sigma: {np.std(data2)}'), row=2, col=1)
fig.add_trace(go.Scatter(y=data3, name=f'Mean: {np.mean(data3)}, Sigma: {np.std(data3)}'), row=3, col=1)

# Update layout for shared x-axis and set titles
fig.update_layout(title_text=f"PTB processed examples")
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_yaxes(title_text="Amplitude")

# Adjust subplot spacing and legend
fig.update_layout(showlegend=True)
fig.update_traces(mode='lines')

# Save the figure as an HTML file
os.makedirs("/home/tzikos/UsefulFigs", exist_ok=True)
fig.write_html(f"/home/tzikos/UsefulFigs/testNorm.html")
'''
