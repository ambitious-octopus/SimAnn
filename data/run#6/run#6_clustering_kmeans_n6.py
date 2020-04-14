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
th_dfd = 0

#Creo una lista degli indici da eliminare
#pick_index_dtw = input_parameter[input_parameter["dtw"] > th_dtw].index.tolist()
pick_index_dfd = np.array(input_parameter[input_parameter["dfd"] > th_dfd].index.tolist())
pick_index_dfd = pick_index_dfd-1
data_th = (data.iloc[np.arange(0,151),pick_index_dfd]).to_numpy().T

#Scalo i parametri
std = StandardScaler()
data_th_sca = std.fit_transform(data_th)


#%%
#CERCO I PARAMETRI MIGLIORI
sillhoute_scores = []
inertia = []
n_cluster_list = np.arange(2, 11).astype(int)
temp_counter = 0


for n_cluster in n_cluster_list:
    kmeans = KMeans(n_clusters=n_cluster, n_init=10, max_iter=300, random_state=23)
    cluster_found = kmeans.fit_predict(data_th_sca)
    sillhoute_scores.append(silhouette_score(data_th_sca, kmeans.labels_))
    inertia.append(kmeans.inertia_)
    temp_counter += 1
    print("n_cluster: " + str(temp_counter))

plt.plot(sillhoute_scores)
plt.show()

#%%
#Faccio il fit della combinazione migliore e creo una lista con l'indice della curva e la rispettiva classe
clusters = 6
kmeans = KMeans(n_clusters=clusters, n_init=10, max_iter=300, random_state=23)
cluster_found = kmeans.fit_predict(data_th)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
centroid = kmeans.cluster_centers_
plt.plot(data_th.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,0], label='0 n= ' + str(cluster_found_sr.value_counts()[0]))
plt.plot(centroid.T[:,1], label='1 n= ' + str(cluster_found_sr.value_counts()[1]))
plt.plot(centroid.T[:,2], label='2 n= ' + str(cluster_found_sr.value_counts()[2]))
plt.plot(centroid.T[:,3], label='3 n= ' + str(cluster_found_sr.value_counts()[3]))
plt.plot(centroid.T[:,4], label='4 n= ' + str(cluster_found_sr.value_counts()[4]))
plt.plot(centroid.T[:,5], label='5 n= ' + str(cluster_found_sr.value_counts()[5]))
plt.plot(topos, label="Topos")
plt.legend(loc="upper left")
plt.show()

#Estraggo i cluster
cluster_1 = data_th[np.where(cluster_found == 0)[0],:]
cluster_2 = data_th[np.where(cluster_found == 1)[0],:]
cluster_3 = data_th[np.where(cluster_found == 2)[0],:]
cluster_4 = data_th[np.where(cluster_found == 3)[0],:]
cluster_5 = data_th[np.where(cluster_found == 4)[0],:]
cluster_6 = data_th[np.where(cluster_found == 5)[0],:]


fig = plt.figure()
plt.subplot(2, 3, 1)
plt.plot(cluster_1.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,0], label='1 n= ' + str(cluster_found_sr.value_counts()[0]))
plt.legend(loc="upper left")
plt.subplot(2, 3, 2)
plt.plot(cluster_2.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,1], label='2 n= ' + str(cluster_found_sr.value_counts()[1]))
plt.legend(loc="upper left")
plt.subplot(2, 3, 3)
plt.plot(cluster_3.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,2], label='3 n= ' + str(cluster_found_sr.value_counts()[2]))
plt.legend(loc="upper left")
plt.subplot(2, 3, 4)
plt.plot(cluster_4.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,3], label='4 n= ' + str(cluster_found_sr.value_counts()[3]))
plt.legend(loc="upper left")
plt.subplot(2, 3, 5)
plt.plot(cluster_5.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,4], label='5 n= ' + str(cluster_found_sr.value_counts()[4]))
plt.legend(loc="upper left")
plt.subplot(2, 3, 6)
plt.plot(cluster_6.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,5], label='6 n= ' + str(cluster_found_sr.value_counts()[5]))
plt.legend(loc="upper left")
plt.show()



#%%
for index, cluster in enumerate(cluster_found):
    if cluster == 1:
        plt.plot(data_th[index], color="grey", alpha=0.1)

plt.plot(centroid.T[:,1], color="red")
plt.show()

#%%
#Scaling
kmeans = KMeans(n_clusters=6)
kmeans.fit(data_th_sca)
data_th_rev = std.inverse_transform(data_th_sca)
cluster_found_sca = kmeans.predict(data_th_rev)
centroid_sca = std.inverse_transform(kmeans.cluster_centers_)
plt.plot(data_th_rev.T, alpha=0.1, color="gray")
plt.plot(centroid_sca.T)
plt.show()


for index, cluster in enumerate(cluster_found_sca):
    if cluster == 3:
        plt.plot(data_th_rev[index], color="grey", alpha=0.1)

plt.plot(centroid.T[:,3], color="red")
plt.show()
