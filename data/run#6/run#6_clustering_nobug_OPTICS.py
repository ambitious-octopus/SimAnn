import pandas as pd
pd.options.mode.chained_assignment = None #Questa funzione disabilità il noiosissimo CopyWarning
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
plt.style.use("seaborn")

# Setto al working directory
base_path = os.getcwd()
data_path = os.path.join(base_path, "data")
run6_path = os.path.join(data_path, "run#6")
os.chdir(run6_path)
os.getcwd()

# %% CARICO I DATI
def reader(file):
    data = pd.read_csv(file, skiprows=6)
    return data
raw_data = reader("run#6-spreadsheet.csv")
# Divido raw_data in tre parti
input_parameter, descriptive, data = raw_data[:12], raw_data[12:19], raw_data[19:]
# Processo gli input_parameter
parameter = input_parameter["[run number]"]
input_parameter.drop(["[run number]"], axis=1, inplace=True)
input_parameter.index = parameter
input_parameter.columns = np.arange(1, 78126)
# Elimino la prima colonna che non indica niente
data = data.drop("[run number]", axis=1)
# Cambio il nome delle righe
new_index = np.arange(0, 151)
data.index = new_index
data = data.astype("float64")
# Carico la media dell'unemployement rate sulle 1000 run
topos_raw = pd.DataFrame(np.loadtxt("media_run").astype("float64"))
# Prendo saolo i primi 150 ticks
topos = topos_raw[:151]
# Carico il vettore con le distanze di frechet
dfd = np.loadtxt("Discrete_Frechet_distance").astype("float64")
# Carico il vettore con il dynamic time warping
dtw = np.loadtxt("Dynamic_Time_Warping").astype("float64")
# Li inserisco nel dataframe dei parametri di input
input_parameter.loc["dfd"] = dfd
input_parameter.loc["dtw"] = dtw
input_parameter = input_parameter.astype("float64")
# Traspongo per lavorarci meglio
data = data.T
input_parameter = input_parameter.T
# Elimino le curve generate dal bug
bug_index = np.array(data[data[148] > 0.90].index.tolist())
data.drop(bug_index, axis=0, inplace=True)
input_parameter.drop(bug_index.astype(int), axis=0, inplace=True)
# %%
# Tolgo i parametri che non sono stati cambiati
unused_parameter = ["production-shock-rho", "price-shock-eta", "wages-shock-xi", "beta", "interest-shock-phi"]
input_parameter.drop(unused_parameter, axis=1, inplace=True)
# Seleziono un valore di metrica e faccio uno slice sulla base di questo
th_dtw = 0
th_dfd = 0.20
# Sualla base della th faccio uno slice dei parametri e delle curve
th_index =  input_parameter['dfd'] > th_dfd
par_th = input_parameter[th_index]
# pick_index = input_parameter[input_parameter["dtw"] > th_dtw].index.tolist()
pick_index = np.array(input_parameter[input_parameter["dfd"] > th_dfd].index.tolist())
pick_index_real = pick_index - 1
data_th = (data.iloc[pick_index_real, np.arange(0, 151)]).to_numpy()
# %%
from sklearn.cluster import OPTICS
optics = OPTICS(min_samples=5)
optics.fit(data_th)
fig = plt.figure()
for index, element in enumerate(np.unique(optics.labels_)):
    sum = np.sum(optics.labels_ == element)
    plt.subplot(np.floor(len(np.unique(optics.labels_))/2), np.floor(len(np.unique(optics.labels_))/2), index + 1)
    plt.plot((data_th[np.where(optics.labels_ == element)[0],:]).T, alpha=0.3, color="gray")
    plt.text(0.2, 0.98, "     c = " + str(element) + " n = " + str(sum),  fontsize=9)
plt.show()