import pandas as pd
pd.options.mode.chained_assignment = None #Questa funzione disabilità il CopyWarning
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
# Faccio il fit della combinazione migliore e creo una lista con l'indice della curva e la rispettiva classe
clusters = 3
kmeans = KMeans(n_clusters=clusters, n_init=15, max_iter=450, random_state=23)
cluster_found = kmeans.fit_predict(data_th)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
centroid = kmeans.cluster_centers_
plt.plot(data_th.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:, 0], label='0 n= ' + str(cluster_found_sr.value_counts()[0]))
plt.plot(centroid.T[:, 1], label='1 n= ' + str(cluster_found_sr.value_counts()[1]))
plt.plot(centroid.T[:, 2], label='2 n= ' + str(cluster_found_sr.value_counts()[2]))
#plt.plot(centroid.T[:, 3], label='3 n= ' + str(cluster_found_sr.value_counts()[3]))
plt.plot(topos, label="Topos")
plt.legend(loc="upper left")
plt.show()

# Estraggo i cluster
cluster_1 = data_th[np.where(cluster_found == 0)[0], :]
cluster_2 = data_th[np.where(cluster_found == 1)[0], :]
cluster_3 = data_th[np.where(cluster_found == 2)[0], :]
#cluster_4 = data_th[np.where(cluster_found == 3)[0], :]

# Faccio dei subplot con tutte le curve
fig = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(cluster_1.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:, 0], label='1 n= ' + str(cluster_found_sr.value_counts()[0]), color="red")
plt.legend(loc="upper left")
plt.subplot(2, 2, 2)
plt.plot(cluster_2.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:, 1], label='2 n= ' + str(cluster_found_sr.value_counts()[1]), color="red")
plt.legend(loc="upper left")
plt.subplot(2, 2, 3)
plt.plot(cluster_3.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:, 2], label='3 n= ' + str(cluster_found_sr.value_counts()[2]), color="red")
plt.legend(loc="upper left")
#plt.subplot(2, 2, 4)
#plt.plot(cluster_4.T, alpha=0.1, color="gray")
#plt.plot(centroid.T[:, 3], label='4 n= ' + str(cluster_found_sr.value_counts()[3]), color="red")
#plt.legend(loc="upper left")
plt.show()


# %%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

#Aggiungo una colonna con il rispettivo cluster
par_th["cluster"] = cluster_found

par_th["number-of-firms"] = np.log(par_th["number-of-firms"])
par_th["dfd"] = np.log(par_th["dfd"])
par_th["dtw"] = np.log(par_th["dtw"])
matrix_scaled_par_reg = StandardScaler().fit_transform(par_th.values)
scaled_par_reg = pd.DataFrame(matrix_scaled_par_reg, index=par_th.index, columns=par_th.columns)
scaled_par_reg.drop("cluster", axis=1, inplace=True)
scaled_par_reg["cluster"] = cluster_found

for cl_id in range(clusters):
    print("cluster", cl_id)
    par_reg_clas = scaled_par_reg[scaled_par_reg['cluster'] == cl_id]
    x = par_reg_clas.iloc[:, 0:7]
    y = par_reg_clas.iloc[:, 8]  # pick dfd
    print("mean dfd = ", np.mean(y))
    print("std dfd = ", np.std(y))
    # Istanzio un Polinomial features, 2°grado
    poly = PolynomialFeatures(2, include_bias=True)
    # Faccio un fit.trasform sulla x
    x_poly = poly.fit_transform(x)
    # Istanzio una regressione lineare
    model = LinearRegression()
    # Faccio il fit
    model.fit(x_poly, y)
    # Estraggo la y predetta
    y_poly_pred = model.predict(x_poly)
    # Calcolo l'erroe
    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    print("rmse: ", rmse)
    # Calcolo r2
    r2 = model.score(x_poly, y)
    print("R^2: ", r2)
    # Estraggo i coefficenti del modello
    coef = model.coef_
    # Estraggo le combinazioni
    coef_names = np.array(poly.get_feature_names())
    # Creo un dataframe con coef e combinazioni
    coef_and_names = pd.DataFrame({"coef": coef, "names": coef_names})
    # Li ordino in modod da visualizzarli meglio
    coef_and_names = coef_and_names.sort_values(by='coef')
    print("head")
    print(coef_and_names.head())
    print("tail")
    print(coef_and_names.tail())
    print("Media Colonne")
    print(np.mean(x))