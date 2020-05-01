import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.style.use("seaborn")

# %%
#Setto al working directory
base_path = os.getcwd()
data_path = os.path.join(base_path, "data")
run6_path = os.path.join(data_path, "run#6")
os.chdir(run6_path)
os.getcwd()
# %%
#Carico i dati
def reader(file):
    data = pd.read_csv(file, skiprows=6)
    return data
raw_data = reader("run#6-spreadsheet-extra.csv")

input_parameter, descriptive, data = raw_data[:12], raw_data[12:19], raw_data[19:]
data = data.drop("[run number]", axis=1)
# Cambio il nome delle righe
new_index = np.arange(0, 151)
data.index = new_index
data = data.astype("float64")


plt.plot(data)
plt.show()
