# -*- coding: utf-8 -*-
"""
Created by Francesco Mattioli
Francesco@nientepanico.org
www.nientepanico.org
"""

#SCARICARE DATABASE QUI 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def reader(file):
    data = pd.read_csv(file, skiprows=6)
    return data

raw_data = reader("run#3-spreadsheet.csv")

#Divido raw_data in due parti: data_head con info descrittive e data con i dati attuali
data_head, data= raw_data[:19], raw_data[19:]

#Creo una lista dei parametri di output
out_parameter = data_head.loc[[12],"1":"1.13"]
out_parameter.index = [0]

#Elimino la prima colonna che non indica una mazza
data = data.drop("[run number]", axis=1)

#Cambio il nome delle righe
new_index = np.arange(0,301)
data.index = new_index

#Funzione che calcola la media per colonna prende come argomenti il dataframe e il punto del parametro
#Esempio, voglio la media di tutte le colonne del parametro .4 devo fare: mean(dataframe, ".4")
def mean(dataframe, col=""):
    mean_list = np.array([])
    for a in range(1,1001):
        column = dataframe[str(a)+col].to_numpy(dtype=np.float64)
        mean = np.mean(column)
        mean_list = np.append(mean_list,mean)
    return mean_list

#Funzione che calcola la deviazione standard
def std(dataframe, col=""):
    std_list = np.array([])
    for a in range(1,1001):
        column = dataframe[str(a)+col]
        column=column.to_numpy(dtype=np.float64)
        std = np.std(column)
        std_list = np.append(std_list,std)
    return std_list

mean_0 = mean(data)
x = np.arange(0,1000)

mean_1 = mean(data,".1")
mean_2 = mean(data, ".2")
mean_3 = mean(data, ".3")


fig = make_subplots(rows=2, 
                    cols=2, subplot_titles=("count workers with [not employed?] / count workers", 
                                            "ln-hopital nominal-GDP", 
                                            "max [production-Y] of fn-incumbent-firms", 
                                            "max [production-Y] of fn-incumbent-firms"),
                    vertical_spacing=0.08, 
                    specs=[[{"type": "scatter"},
                            {"type": "scatter"}],
                            [{"type": "scatter"},
                            {"type": "scatter"}]])

fig.add_trace(go.Scatter(x=x, y=mean_0,mode="lines",name="1"),row=1, col=1)

fig.add_trace(go.Scatter(x=x,y=mean_1,mode="lines",name="1.1"),row=1, col=2)

fig.add_trace(go.Scatter(x=x, y=mean_2,mode="lines",name="1.2"),row=2, col=1)

fig.add_trace(go.Scatter(x=x, y=mean_3,mode="lines",name="1.3"),row=2, col=2)

fig.update_layout(title = 'Mean per run')


plot(fig)


