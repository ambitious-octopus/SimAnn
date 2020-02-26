# -*- coding: utf-8 -*-
"""
Created by Francesco Mattioli
Francesco@nientepanico.org
www.nientepanico.org
"""

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

raw_data = reader("run#5-spreadsheet.csv")

#Divido raw_data in due parti: data_head con info descrittive e data con i dati attuali
data_head, data= raw_data[:18], raw_data[18:]

#Elimino la prima colonna che non indica una mazza
data = data.drop("[run number]", axis=1)

#Creo una lista dei parametri di output
out_parameter = data_head.loc[[12],"1":"1.20"]

#Cambio il nome delle righe
new_index = np.arange(0,302)
data.index = new_index

out_parameter.index = [0]
data = data.drop(0, axis=0)

data = data.astype("float64")


#%%
#FUNZIONI DI PROCESSING

#Funzione che prende come parametro un dataset e un tick_window e ritorna un dataframe con la window richiesta
def window_selector(data,tick_window):
    window = data.drop(data.index[:-tick_window-1])
    return window
    
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
def std(dataframe, par=""):
    std_list = np.array([])
    for a in range(1,1001):
        column = dataframe[str(a)+par]
        column=column.to_numpy(dtype=np.float64)
        std = np.std(column)
        std_list = np.append(std_list,std)
    return std_list

#Funzione che dato un dataset lo splitta in base al numero della run e al parametro
def column_splitter(data, num_run, par=""):
    lista = []
    for a in range(1,num_run+1):
        lista.append(str(a)+par)
    colonne = data[lista]
    return colonne

#Meglio che non lo scrivo cosa fa, lo so, è bruttissima
def error_window(series):
    a_1 = np.arange(0,25)
    a_2 = np.arange(26,50)
    a_3 = np.arange(51,75)
    a_4 = np.arange(76, 100)
    a_5 = np.arange(101, 125)
    a_6 = np.arange(126, 150)
    a_7 = np.arange(151, 175)
    a_8 = np.arange(176, 200)
    a_9 = np.arange(201, 225)
    a_10 = np.arange(226, 250)
    a_11 = np.arange(251, 275)
    a_12 =np.arange(276, 302)
    lista = (np.concatenate((a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9,a_10,a_11,a_12), axis=None)).tolist()
    for index, value in series.items():
        if index in lista:
            series[index] = "null"
    return series

#%%
#CALCOLO GLI INDICATORI SULLE DIMESIONI PRESE IN ESAME

# count workers with [not employed?] / count workers
unemp_rate = column_splitter(data, 1000, par="")
unemp_rate_mean = unemp_rate.mean(axis=1)
unemp_rate_median = unemp_rate.median(axis=1)
unemp_rate_max = unemp_rate.max(axis=1)
unemp_rate_min = unemp_rate.min(axis=1)
unemp_rate_std = error_window(unemp_rate.std(axis=1))


# ln-hopital nominal-GDP
nominal_GDP = column_splitter(data, 1000, par=".1")
nominal_GDP_mean = nominal_GDP.mean(axis=1)
nominal_GDP_median = nominal_GDP.median(axis=1)
nominal_GDP_max = nominal_GDP.max(axis=1)
nominal_GDP_min = nominal_GDP.min(axis=1)
nominal_GDP_std = error_window(nominal_GDP.std(axis=1))

# mean [production-Y] of fn-incumbent-firms
production_inc_firm = column_splitter(data, 1000, par=".4")
production_inc_firm_mean = production_inc_firm.mean(axis=1)
production_inc_firm_median = production_inc_firm.median(axis=1)
production_inc_firm_max = production_inc_firm.max(axis=1)
production_inc_firm_min = production_inc_firm.min(axis=1)
production_inc_firm_std = error_window(production_inc_firm.std(axis=1))
production_inc_firm_max_real = (column_splitter(data, 1000, par=".3")).max(axis=1)
production_inc_firm_min_real = (column_splitter(data, 1000, par=".5")).min(axis=1)

# mean [production-Y] of firms
production_firm = column_splitter(data, 1000, par=".7")
production_firm_mean = production_firm.mean(axis=1)
production_firm_median = production_firm.median(axis=1)
production_firm_max = production_firm.max(axis=1)
production_firm_min = production_firm.min(axis=1)
production_firm_std = error_window(production_firm.std(axis=1))
production_firm_max_real = (column_splitter(data, 1000, par=".6")).max(axis=1)
production_firm_min_real = (column_splitter(data, 1000, par=".8")).min(axis=1)

# mean [wage-offered-Wb] of firms
wage = column_splitter(data, 1000, par=".9")
wage_mean = wage.mean(axis=1)
wage_median = wage.median(axis=1)
wage_max = wage.max(axis=1)
wage_min = wage.min(axis=1)
wage_std = error_window(wage.std(axis=1))
wage_max_real = (column_splitter(data, 1000, par=".11")).max(axis=1)
wage_min_real = (column_splitter(data, 1000, par=".10")).min(axis=1)

# ln-hopital mean [wealth] of workers
wealth = column_splitter(data, 1000, par=".12")
wealth_mean = wealth.mean(axis=1)
wealth_median = wealth.median(axis=1)
wealth_max = wealth.max(axis=1)
wealth_min = wealth.min(axis=1)
wealth_std = error_window(wealth.std(axis=1))
wealth_max_real = (column_splitter(data, 1000, par=".13")).max(axis=1)
wealth_min_real = (column_splitter(data, 1000, par=".14")).min(axis=1)

# 100 * mean [my-interest-rate] of firms
mult_interest_rate = column_splitter(data, 1000, par=".15")
mult_interest_rate_mean = mult_interest_rate.mean(axis=1)
mult_interest_rate_median = mult_interest_rate.median(axis=1)
mult_interest_rate_max = mult_interest_rate.max(axis=1)
mult_interest_rate_min = mult_interest_rate.min(axis=1)
mult_interest_rate_std = error_window(mult_interest_rate.std(axis=1))
mult_interest_rate_max_real = (column_splitter(data, 1000, par=".17")).max(axis=1)
mult_interest_rate_min_real = (column_splitter(data, 1000, par=".16")).min(axis=1)

# mean [my-interest-rate] of firms
interest_rate = column_splitter(data, 1000, par=".18")
interest_rate_mean = interest_rate.mean(axis=1)
interest_rate_median = interest_rate.median(axis=1)
interest_rate_max = interest_rate.max(axis=1)
interest_rate_min = interest_rate.min(axis=1)
interest_rate_std = error_window(interest_rate.std(axis=1))
interest_rate_max_real = (column_splitter(data, 1000, par=".19")).max(axis=1)
interest_rate_min_real = (column_splitter(data, 1000, par=".20")).min(axis=1)



#%%
# GRAFICO

x= np.arange(0,301)

fig = make_subplots(rows=8, cols=1, 
                    vertical_spacing=0.019, 
                    specs=[[{"type": "scatter"}],
                           [{"type": "scatter"}],
                           [{"type": "scatter"}],
                           [{"type": "scatter"}],
                           [{"type": "scatter"}],
                           [{"type": "scatter"}],
                           [{"type": "scatter"}],
                           [{"type": "scatter"}]], x_title="tick", subplot_titles=("Unemployment rate", 
                                                                                  "Nominal GDP", 
                                                                                  "Production of incumbent-firms",
                                                                                  "Production of incumbent-firms",
                                                                                  "wage-offered-Wb",
                                                                                  "Wealth of workers",
                                                                                  "Contractual interest rate",
                                                                                  "100*Contractual interest rate"))

# count workers with [not employed?] / count workers
fig.add_trace(go.Scatter(x=x,y=unemp_rate_max,name="max",line=dict(width=0.8)), row=1, col=1)
fig.add_trace(go.Scatter(x=x,y=unemp_rate_median,name="median", line=dict(width=1.5), error_y=dict(type='data', array=unemp_rate_std, symmetric=True)),row=1, col=1)
fig.add_trace(go.Scatter(x=x,y=unemp_rate_min,name="min",line=dict(width=0.8)), row=1, col=1)


# ln-hopital nominal-GDP
fig.add_trace(go.Scatter(x=x,y=nominal_GDP_max,name="max",line=dict(width=0.8)), row=2, col=1)
fig.add_trace(go.Scatter(x=x,y=nominal_GDP_median,name="median", line=dict(width=1.5), error_y=dict(type='data', array=nominal_GDP_std, symmetric=True)),row=2, col=1)
fig.add_trace(go.Scatter(x=x,y=nominal_GDP_min,name="min",line=dict(width=0.8)), row=2, col=1)

# mean [production-Y] of fn-incumbent-firms
fig.add_trace(go.Scatter(x=x,y=production_inc_firm_max_real,name="max",line=dict(width=0.8)), row=3, col=1)
fig.add_trace(go.Scatter(x=x,y=production_inc_firm_median,name="median", line=dict(width=1.5), error_y=dict(type='data', array=production_inc_firm_std, symmetric=True)),row=3, col=1)
fig.add_trace(go.Scatter(x=x,y=production_inc_firm_min_real,name="min",line=dict(width=0.8)), row=3, col=1)


# mean [production-Y] of firms
fig.add_trace(go.Scatter(x=x,y=production_firm_max_real,name="max",line=dict(width=0.8)), row=4, col=1)
fig.add_trace(go.Scatter(x=x,y=production_firm_median,name="median", line=dict(width=1.5),  error_y=dict(type='data', array=production_firm_std, symmetric=True)),row=4, col=1)
fig.add_trace(go.Scatter(x=x,y=production_firm_min_real,name="min",line=dict(width=0.8)), row=4, col=1)


# mean [wage-offered-Wb] of firms
fig.add_trace(go.Scatter(x=x,y=wage_max_real,name="max",line=dict(width=0.8)), row=5, col=1)
fig.add_trace(go.Scatter(x=x,y=wage_median,name="median", line=dict(width=1.5), error_y=dict(type='data', array=wage_std, symmetric=True)),row=5, col=1)
fig.add_trace(go.Scatter(x=x,y=wage_min_real,name="min",line=dict(width=0.8)), row=5, col=1)


# ln-hopital mean [wealth] of workers
fig.add_trace(go.Scatter(x=x,y=wealth_max_real,name="max",line=dict(width=0.8)), row=6, col=1)
fig.add_trace(go.Scatter(x=x,y=wealth_median,name="median", line=dict(width=1.5),  error_y=dict(type='data', array=wealth_std, symmetric=True)),row=6, col=1)
fig.add_trace(go.Scatter(x=x,y=wealth_min_real,name="min",line=dict(width=0.8)), row=6, col=1)


# 100 * mean [my-interest-rate] of firms
fig.add_trace(go.Scatter(x=x,y=mult_interest_rate_max_real,name="max",line=dict(width=0.8)), row=7, col=1)
fig.add_trace(go.Scatter(x=x,y=mult_interest_rate_median,name="median", line=dict(width=1.5),  error_y=dict(type='data', array=mult_interest_rate_std, symmetric=True)),row=7, col=1)
fig.add_trace(go.Scatter(x=x,y=mult_interest_rate_min_real,name="min",line=dict(width=0.8)), row=7, col=1)


# mean [my-interest-rate] of firms
fig.add_trace(go.Scatter(x=x,y=interest_rate_max_real,name="max",line=dict(width=0.8)), row=8, col=1)
fig.add_trace(go.Scatter(x=x,y=interest_rate_median,name="median", line=dict(width=1.5), error_y=dict(type='data', array=interest_rate_std, symmetric=True)),row=8, col=1)
fig.add_trace(go.Scatter(x=x,y=interest_rate_min_real,name="min",line=dict(width=0.8)), row=8, col=1)



fig.update_layout(height=2600, width=1600, showlegend=False)
plot(fig)





















