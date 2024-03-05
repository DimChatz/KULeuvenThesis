import pandas as pd
import numpy as np
from preprocessing import downsampler, noiseRemover, createMissingLeads
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def preprocVis():
    filePath = "/home/tzikos/Desktop/Data/Berts/AVNRT/AVNRT tachy/AVNRT tachy/75E6EFE9-4EC8-4B3A-9E89-5004E6A326F2_2AX1ZWAX3DRF_20220426_2.xlsx"
    data = pd.read_excel(filePath)
    data = downsampler(data)
    orig = data[data.columns[1]].to_numpy()
    data = data.drop(data.columns[0], axis=1)
    data = data.to_numpy()
    mean, sigma = 0.2697284861743545, 634.6015221068606
    normData = (data - mean) / sigma
    procData = noiseRemover(normData)
    procData = np.expand_dims(procData, axis = -1)
    procData = createMissingLeads(procData)

    # Generating data
    x = np.linspace(0, 3, 750)
    y1 = orig
    y2 = normData[:750,0]
    y3 = procData[:750,0,0]

    # Create figure with 2 subplots arranged in 1 row and 2 columns
    fig = make_subplots(rows=3, cols=1)


    # Adding scatter plot to the first subplot
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='noise Removed'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='noise Removed'), row=3, col=1)


    # Updating layout (optional)
    fig.update_layout(title_text="before and after")

    # Showing the plot
    fig.show()


def trainVisualizer(trainLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List):
    # Create figure with 2 subplots arranged in 1 row and 2 columns
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Loss", "Accuracy", "F1 Score"))
    # Adding scatter plot to the first subplot
    x = np.linspace(1, len(trainLossList))
    fig.add_trace(go.Scatter(x=x, y=trainLossList, mode='lines', name='train loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=valLossList, mode='lines', name='val loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=trainAccList, mode='lines', name='train Acc'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=valAccList, mode='lines', name='val Acc'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=trainF1List, mode='lines', name='train F1'), row=1, col=3)
    fig.add_trace(go.Scatter(x=x, y=valF1List, mode='lines', name='val F1'), row=1, col=3)
    # Updating layout (optional)
    fig.update_layout(
        title_text="Training Visualization",
        title_x=0.5,  # Centers the main title
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.show()