import pandas as pd
import numpy as np
from preprocessing import noiseRemover, createMissingLeads
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio


def trainVisualizer(trainLossList, valLossList, trainAccList, valAccList, trainF1List, valF1List, saveName):
    """Function to visualize the training process"""
    # Create figure with 2 subplots arranged in 1 row and 3 columns
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Loss", "Accuracy", "F1 Score"))
    # Adding scatter plot to the first subplot
    xAxis = np.linspace(1, len(trainLossList), len(trainLossList))
    fig.add_trace(go.Scatter(x=xAxis, y=trainLossList, mode='lines', name='train loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=xAxis, y=valLossList, mode='lines', name='val loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=xAxis, y=trainAccList, mode='lines', name='train Acc'), row=1, col=2)
    fig.add_trace(go.Scatter(x=xAxis, y=valAccList, mode='lines', name='val Acc'), row=1, col=2)
    fig.add_trace(go.Scatter(x=xAxis, y=trainF1List, mode='lines', name='train F1'), row=1, col=3)
    fig.add_trace(go.Scatter(x=xAxis, y=valF1List, mode='lines', name='val F1'), row=1, col=3)
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
    # Saving the plot to an html file
    fig.write_html(f"/home/tzikos/histories{saveName}.html")
    fig.show()



def VisNP(npArray, saveName, comment=" "):
    """Function to visualize PTBXL data at each step of the process"""
    # Generating data
    x = np.linspace(0, 1, 5000)
    y1 = npArray[:, 0]
    y2 = npArray[:, 1]

    # Create figure with 2 subplots arranged in 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2)
    # Adding scatter plot to the first subplot
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Lead 1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Lead 2'), row=1, col=2)
    # Updating layout
    fig.update_layout(title_text=f"{comment}")
    # Showing the plot
    fig.write_html(f"/home/tzikos/{saveName}.html")
    fig.show()

def Vis(filePath1, filePath2, saveName, comment=" "):
    """Function to visualize PTBXL data at each step of the process"""
    # Generating data
    x = np.linspace(0, 10, 2500)
    y1 = np.load(filePath1)
    y1 = y1[:, 0]
    y2 = np.load(filePath2)
    y2 = y2[:, 0]

    # Create figure with 2 subplots arranged in 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2)
    # Adding scatter plot to the first subplot
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='AVNRT'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Normal'), row=1, col=2)
    # Updating layout
    fig.update_layout(title_text=f"{comment}")
    # Showing the plot
    fig.write_html(f"/home/tzikos/{saveName}.html")
    fig.show()

def plotNSaveConfusion(cm, classNames, saveStr, text):
    """
    Plot and save a confusion matrix using Plotly with row-wise color intensity and integer annotations.

    Parameters:
    - cm: Confusion matrix (2D numpy array).
    - classNames: List of class names (labels).
    - saveStr: Name of the file to save the plot.
    - text: Title text for the confusion matrix.
    """
    # Normalize each row by its maximum value for color scaling
    cm_normalized = np.array([row / np.sum(row) for row in cm])
    
    # Create annotations (text in each cell) using the original integer values from cm
    annotations = [[str(int(value)) for value in row] for row in cm]

    # Define the figure using a heatmap for the confusion matrix with integer annotations
    fig = ff.create_annotated_heatmap(z=cm_normalized, x=classNames, y=classNames, annotation_text=annotations, colorscale='Magma')

    # Update layout to make it more readable
    fig.update_layout(title_text=f"{text} Confusion Matrix",
                      xaxis=dict(title='Predicted value'),
                      yaxis=dict(title='Actual value'),
                      yaxis_autorange='reversed',  # This correctly flips the y-axis to match conventional confusion matrix layout
                      xaxis_tickangle=-45)
    # Show figure in the notebook or IDE
    #fig.show()
    # Save the figure as an HTML file
    fig.write_html(f"/home/tzikos/{saveStr}.html")
    return fig

def dataLeadStatsVis(filePath="/home/tzikos/TableCreatorVis"):
    typeExp = ["pre", "tachy"]
    expList = ['normal', 'AVNRT', 'AVRT', 'concealed', 'EAT']
    
    for taip in typeExp:
        data = pd.read_csv(f"{filePath}/{taip}VisTable.csv")
        subplot_titles = []
        
        for clasS in expList:
            for i in range(12):
                dataTemp = data[(data['Class'] == clasS) & (data['Lead'] == i+1)]
                mean_of_means = dataTemp['Mean'].mean()
                sigma_of_means = dataTemp['Mean'].std()
                mean_of_sigmas = dataTemp['Sigma'].mean()
                sigma_of_sigmas = dataTemp['Sigma'].std()
                subplot_titles.append(f'Lead {i+1} - MM:{mean_of_means:.2f}, SM:{sigma_of_means:.2f}, MS:{mean_of_sigmas:.2f}, SS:{sigma_of_sigmas:.2f}')
            
            # Create figure with dynamic subplot titles
            fig = make_subplots(rows=3, cols=4, subplot_titles=subplot_titles)
            
            for i in range(12):
                dataTemp = data[(data['Class'] == clasS) & (data['Lead'] == i+1)]
                for dtype, color in zip(['Train', 'Val'], ['#33BEFF', '#DE4A0E']):
                    x = np.array(dataTemp[dataTemp['Dataset'] == dtype]['Mean'])
                    y = np.array(dataTemp[dataTemp['Dataset'] == dtype]['Sigma'])
                    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{dtype} Lead {i+1}', 
                                             marker=dict(color=color, size=6)),
                                  row=i // 4 + 1, col=i % 4 + 1)
            
            fig.update_layout(title_text=f"For Class {clasS} in {taip}")
            fig.write_html(f"{filePath}/{clasS}_{taip}.html")



def dataLeadStatsVisPerLead(filePath="/home/tzikos/TableCreatorVis"):
    typeExp = ["pre", "tachy"]
    expList = ['normal', 'AVNRT', 'AVRT', 'concealed', 'EAT']
    
    for taip in typeExp:
        data = pd.read_csv(f"{filePath}/{taip}VisTable.csv")
        subplot_titles = []
        
        for clasS in expList:
            for i in range(12):
                # Create figure with dynamic subplot titles
                fig = make_subplots(rows=1, cols=3, subplot_titles=subplot_titles)
                for idx, (dtype, color) in enumerate(zip(['Train', 'Val', 'Test'], ['#33BEFF', '#DE4A0E', '#FFA500'])):
                    dataTemp = data[(data['Class'] == clasS) & (data['Lead'] == i+1) & (data['Dataset'] == dtype)]
                    mean_of_means = dataTemp['Mean'].mean()
                    sigma_of_means = dataTemp['Mean'].std()
                    mean_of_sigmas = dataTemp['Sigma'].mean()
                    sigma_of_sigmas = dataTemp['Sigma'].std()
                    subplot_titles.append(f'{dtype} Lead {i+1} - MM:{mean_of_means:.2f}, SM:{sigma_of_means:.2f}, MS:{mean_of_sigmas:.2f}, SS:{sigma_of_sigmas:.2f}')
                    x = np.array(dataTemp['Mean'])
                    y = np.array(dataTemp['Sigma'])
                    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f'{dtype} Lead {i+1}', 
                                             marker=dict(color=color, size=6)),
                                  row=1, col=idx+1)
                fig.update_layout(title_text=f"For Class {clasS} in {taip}")
                fig.write_html(f"{filePath}/{clasS}_{taip}_Lead{i+1}.html")
