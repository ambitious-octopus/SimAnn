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

#Elimino la prima colonna che non indica una mazza
data = data.drop("[run number]", axis=1)

#Cambio il nome delle righe
new_index = np.arange(0,151)
data.index = new_index

data = data.astype("float64")
#%%

y_dato = data["1"].to_numpy(np.float64).reshape((151,1))

dato = np.hstack((x_dato,y_dato))


freschetto = sm.frechet_dist(topos, dato)
#%%
scale_factor = 150
time = new_index/scale_factor
x = time.reshape((151,1))
#Carico la media della run#5 per Unemployement Rate
raw_media_run = np.loadtxt("media_run").astype("float64")
#Faccio uno slicing dei primi e un reshape
media_run = raw_media_run[:151].reshape((151,1))

topos = topos = np.hstack((x,media_run))

Discrete_Frechet_distance = []
Curve_Length_based = []
start = time.time()
for a in range(1,100):
    y = data[str(a)].to_numpy(np.float64).reshape((151,1))
    cord_new = np.hstack((x,y))
    Discrete_Frechet_distance.append(sm.frechet_dist(topos,cord_new))
    print("Column: " + str(a))
    end = time.time()
    print("Tempo di processamento: " + str(end - start))







