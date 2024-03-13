import pandas as pd
import numpy as np
from preprocessing import downsampler, noiseRemover, createMissingLeads
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def preprocVis(filePath, mean, sigma):
    data = pd.read_excel(filePath)
    data = downsampler(data)
    orig = data[data.columns[1]].to_numpy()
    data = data.drop(data.columns[0], axis=1)
    data = data.to_numpy()
    normData = (data - mean) / sigma
    procData = noiseRemover(normData)
    procData = np.expand_dims(procData, axis = -1)
    procData = createMissingLeads(procData)

    # Generating data
    x = np.linspace(0, 3, 750)
    y1 = orig
    y2 = normData[:750,0]
    y3 = procData[:750,0,0]
    y4 = procData[:750,0,1]

    # Create figure with 2 subplots arranged in 1 row and 2 columns
    fig = make_subplots(rows=2, cols=2)


    # Adding scatter plot to the first subplot
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='normalized data'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='fully processed Lead'), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=y4, mode='lines', name='should-be-missing lead'), row=2, col=2)
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



def VisNP(npArray, saveName, comment=" "):
    # Generating data
    x = np.linspace(0, 1, 5000)
    y1 = npArray[:, 0]
    y2 = npArray[:, 1]

    # Create figure with 2 subplots arranged in 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2)

    # Adding scatter plot to the first subplot
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Lead 1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Lead 2'), row=1, col=2)
    # Updating layout (optional)
    fig.update_layout(title_text=f"{comment}")
    # Showing the plot
    fig.write_html(f"/home/tzikos/{saveName}.html")
    fig.show()
