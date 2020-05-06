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
#%%
# Elimino le curve generate dal bug
data_pp = data.to_numpy()
index_bug = []
for index,curve in enumerate(data_pp):
    grad = np.gradient(curve)
    counter = 0
    if grad[140] == grad[141] and grad[141] == grad[142] and grad[142] == grad[143]:
        index_bug.append(index)
        print(grad[140:143])
for i in index_bug:
    plt.plot(data_pp[i])
plt.show()
print(len(index_bug))


index_bug_s = [str(x+1) for x in index_bug]
index_bug = [x+1 for x in index_bug]
data.drop(index_bug_s, axis=0, inplace=True)
input_parameter.drop(index_bug, axis=0, inplace=True)
# %%
# Tolgo i parametri che non sono stati cambiati
unused_parameter = ["production-shock-rho", "price-shock-eta", "wages-shock-xi", "beta", "interest-shock-phi"]
input_parameter.drop(unused_parameter, axis=1, inplace=True)
#%%
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
#%%
#CERCO I PARAMETRI MIGLIORI
sillhoute_scores = []
inertia = []
n_cluster_list = np.arange(2, 8).astype(int)
temp_counter = 0

for n_cluster in n_cluster_list:
    kmeans = KMeans(n_clusters=n_cluster, n_init=15, max_iter=450, random_state=23)
    cluster_found = kmeans.fit_predict(data)
    sillhoute_scores.append(silhouette_score(data, kmeans.labels_))
    inertia.append(kmeans.inertia_)
    temp_counter += 1
    print("n_cluster: " + str(temp_counter))

plt.plot(n_cluster_list,sillhoute_scores, label='silhoute')
plt.show()
plt.plot(n_cluster_list,inertia, label='inertia')
plt.show()
#%%
clusters = 4
kmeans = KMeans(n_clusters=clusters, n_init=15, max_iter=450, random_state=23)
cluster_found = kmeans.fit_predict(data_th)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
centroid = kmeans.cluster_centers_
plt.plot(data_th.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,0], label='0 n= ' + str(cluster_found_sr.value_counts()[0]))
plt.plot(centroid.T[:,1], label='1 n= ' + str(cluster_found_sr.value_counts()[1]))
plt.plot(centroid.T[:,2], label='2 n= ' + str(cluster_found_sr.value_counts()[2]))
plt.plot(centroid.T[:,3], label='3 n= ' + str(cluster_found_sr.value_counts()[3]))
plt.plot(topos, label="Topos")
plt.legend(loc="upper left")
plt.show()

#Estraggo i cluster
cluster_1 = data_th[np.where(cluster_found == 0)[0],:]
cluster_2 = data_th[np.where(cluster_found == 1)[0],:]
cluster_3 = data_th[np.where(cluster_found == 2)[0],:]
cluster_4 = data_th[np.where(cluster_found == 3)[0],:]

#Faccio dei subplot con tutte le curve
fig = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(cluster_1.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,0], label='1 n= ' + str(cluster_found_sr.value_counts()[0]), color="red")
plt.legend(loc="upper left")
plt.subplot(2, 2, 2)
plt.plot(cluster_2.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,1], label='2 n= ' + str(cluster_found_sr.value_counts()[1]), color="red")
plt.legend(loc="upper left")
plt.subplot(2, 2, 3)
plt.plot(cluster_3.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,2], label='3 n= ' + str(cluster_found_sr.value_counts()[2]), color="red")
plt.legend(loc="upper left")
plt.subplot(2, 2, 4)
plt.plot(cluster_4.T, alpha=0.1, color="gray")
plt.plot(centroid.T[:,3], label='4 n= ' + str(cluster_found_sr.value_counts()[3]), color="red")
plt.legend(loc="upper left")
plt.show()

#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import precision_score, recall_score

par_th["cluster"] = cluster_found


x_raw = par_th.iloc[:,:7]
y_raw = par_th.iloc[:,9]
X_train, X_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw)
rand_clas = RandomForestClassifier(n_estimators=500, random_state=42)
rand_clas.fit(X_train, y_train)
y_pred = rand_clas.predict(X_test)
importance = sorted(zip(rand_clas.feature_importances_, X_train.columns), reverse=True)
accuracy_score(y_test, y_pred)

#Random Grid Search
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_

rtf_2 = RandomForestClassifier(n_estimators=800,min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=100,bootstrap = False)
rtf_2.fit(X_train,y_train)
y_pred_rtf_2 = rtf_2.predict(X_test)
accuracy_score(y_test, y_pred_rtf_2)
importance = sorted(zip(rtf_2.feature_importances_, X_train.columns), reverse=True)

print("Accuracy: ", accuracy_score(y_test, y_pred_rtf_2))
print("Features importance: ")
for e in importance:
    print(e)


#%%
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(X_train,y_train)
y_pred_sgd = kn.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred_sgd))



