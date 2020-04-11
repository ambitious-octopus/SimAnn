import pandas as pd
import numpy as np
import similaritymeasures as sm
import matplotlib.pyplot as plt

from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import os

from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

# %%
base_path = os.getcwd()
data_path = os.path.join(base_path, "data")
run6_path = os.path.join(data_path, "run#6")
os.chdir(run6_path)
os.getcwd()


# %%
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
# Carico i dati
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
for ind, a in enumerate(dtw):
    if a > 100:
        plt.plot(data[str(ind + 1)])
        indici.append(ind + 1)
plt.show()
bad_para = input_parameter[indici]
# %%
input_parameter = input_parameter.T
# Tolgo i parametri che non sono stati cambiati
unused_parameter = ["production-shock-rho", "price-shock-eta", "wages-shock-xi", "beta", "interest-shock-phi"]
for par in unused_parameter:
    input_parameter.drop(par, axis=1, inplace=True)

#Seleziono una misurazione "dfd" o "dtw"
pick = "dfd"


if pick == "dfd":
    a = "dfd"
    b = "dtw"
    threshold = 0.2

else:
    a = "dtw"
    b = "dfd"
    threshold = 15

#Creo una lista degli indici da eliminare
pick_index = input_parameter[input_parameter[a] < threshold].index.tolist()
#Li elimino
input_parameter.drop(pick_index, inplace=True)
#Droppo la colonna con la misura che non mi interessa
input_parameter.drop(b, axis=1, inplace=True)
#Trasformo la colonna number of firms
input_parameter["number-of-firms"] = np.log(input_parameter["number-of-firms"])
input_parameter[a] = np.log(input_parameter[a])

# %%
# Scalo i dati
matrix_scaled_input_parameter = StandardScaler().fit_transform(input_parameter.values)
scaled_input_parameter = pd.DataFrame(matrix_scaled_input_parameter, index=input_parameter.index,
                                      columns=input_parameter.columns)
# %%
# Definisco la x, l'insieme dei parametri
x = matrix_scaled_input_parameter[:, 0:7]
# Definisco la y, la distanza di frechet
y = matrix_scaled_input_parameter[:, 7]
# Istanzio un Polinomial features
poly = PolynomialFeatures(8, include_bias=False)
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
# Calcolo r2
r2 = model.score(x_poly, y)
#Estraggo i coefficenti del modello
coef = model.coef_
#Estraggo le combinazioni
coef_names = np.array(poly.get_feature_names())
#Creo un dataframe con coef e combinazioni
coef_and_names = pd.DataFrame({"coef": coef, "names": coef_names})
#Li ordino in modod da visualizzarli meglio
coef_and_names = coef_and_names.sort_values(by='coef')

x_sel = x[0:,3]

plt.plot(y)
plt.plot(y_poly_pred)
plt.show()
