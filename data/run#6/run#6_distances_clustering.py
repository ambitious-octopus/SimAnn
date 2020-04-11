import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import pyplot as plt
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
raw_data = reader("run#6-spreadsheet.csv")
# Divido raw_data in due parti: data_head con info descrittive e data con i dati attuali
input_parameter, descriptive, data = raw_data[:12], raw_data[12:19], raw_data[19:]
# Processo gli input_parameter
parameter = input_parameter["[run number]"]
input_parameter.index = parameter
input_parameter = input_parameter.drop(["[run number]"], axis=1)
input_parameter.columns = np.arange(1, 78126)
# Elimino la prima colonna che non indica una mazza
data = data.drop("[run number]", axis=1)
# Cambio il nome delle righe
new_index = np.arange(0, 151)
data.index = new_index
data = data.astype("float64")

# %%
# Carico la media dell'unemployement rate sulle 1000 run
topos_raw = pd.DataFrame(np.loadtxt("media_run").astype("float64"))
# Prendo saolo i primi 150 ticks
topos = topos_raw[:151]
# Carico il vettore con le distanze di frechet
dfd = np.loadtxt("Discrete_Frechet_distance").astype("float64")
# Carico il vettore con il dynamic time warping
dtw = np.loadtxt("Dynamic_Time_Warping").astype("float64")
# Li inserisco in un dataframe
input_parameter.loc["dfd"] = dfd
input_parameter.loc["dtw"] = dtw
input_parameter = input_parameter.astype("float64")
indici = []

#for ind, a in enumerate(dtw):
    #if a > 100:
        #plt.plot(data[str(ind + 1)])
        #indici.append(ind + 1)

bad_para = input_parameter[indici]
# %%
input_parameter = input_parameter.T
# Tolgo i parametri che non sono stati cambiati
unused_parameter = ["production-shock-rho", "price-shock-eta", "wages-shock-xi", "beta", "interest-shock-phi"]
for par in unused_parameter:
    input_parameter.drop(par, axis=1, inplace=True)

th_dtw = 40
th_dfd = 0.2

#Creo una lista degli indici da eliminare
#pick_index_dtw = input_parameter[input_parameter["dtw"] > th_dtw].index.tolist()
pick_index_dfd = np.array(input_parameter[input_parameter["dfd"] > th_dfd].index.tolist())
pick_index_dfd = pick_index_dfd-1
data_th = (data.iloc[np.arange(0,151),pick_index_dfd]).to_numpy().T

#%%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sillhoute_scores = []
inertia = []
n_cluster_list = np.arange(2, 18).astype(int)


for n_cluster in n_cluster_list:
    kmeans = KMeans(n_clusters=n_cluster, n_init=15, max_iter=500, random_state=42)
    cluster_found = kmeans.fit_predict(data_th)
    sillhoute_scores.append(silhouette_score(data_th, kmeans.labels_))
    inertia.append(kmeans.inertia_)
#%%
#Faccio il fit della combinazione migliore e creo una lista con l'indice della curva e la rispettiva classe
kmeans = KMeans(n_clusters=4)
cluster_found = kmeans.fit_predict(data_th)
cluster_found_sr = pd.Series(cluster_found, name='cluster')

centroid = kmeans.cluster_centers_

plt.plot(data_th.T, alpha=0.1, color="gray")
plt.plot(centroid.T)
plt.show()


for index, cluster in enumerate(cluster_found):
    if cluster == 1:
        plt.plot(data_th[index], color="grey", alpha=0.1)

plt.plot(centroid.T[:,1], color="red")
plt.show()

plt.plot(inertia)
plt.show()