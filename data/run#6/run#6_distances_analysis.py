import pandas as pd
import numpy as np
import similaritymeasures as sm
import matplotlib.pyplot as plt

from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import os
import time

#%%
base_path = os.getcwd()
data_path = os.path.join(base_path, "data")
run6_path = os.path.join(data_path, "run#6")
os.chdir(run6_path)
os.getcwd()


#%%
def reader(file):
    data = pd.read_csv(file, skiprows=6)
    return data

raw_data = reader("run#6-spreadsheet.csv")

#Divido raw_data in due parti: data_head con info descrittive e data con i dati attuali
input_parameter, descriptive, data = raw_data[:12], raw_data[12:19], raw_data[19:]

#Processo gli input_parameter
parameter = input_parameter["[run number]"]
input_parameter.index = parameter
input_parameter = input_parameter.drop(["[run number]"], axis=1)
input_parameter.columns = np.arange(1,78126)

#Elimino la prima colonna che non indica una mazza
data = data.drop("[run number]", axis=1)

#Cambio il nome delle righe
new_index = np.arange(0,151)
data.index = new_index

data = data.astype("float64")
#%%

topos_raw = pd.DataFrame(np.loadtxt("media_run").astype("float64"))
topos = topos_raw[150:]
input_parameter.loc["dfd"] = np.loadtxt("Discrete_Frechet_distance").astype("float64")
input_parameter.loc["dtw"] = np.loadtxt("Dynamic_Time_Warping").astype("float64")









