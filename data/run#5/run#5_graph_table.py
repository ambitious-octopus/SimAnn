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
from scipy import stats


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
def col_window_selector(data,tick_window):
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

def wind_selector(serie,from_tick,to_tick):
    window = serie.iloc[from_tick:to_tick+1]
    return window


#%%
#PARAMETRI PER IL CALCOLO DEGLI INDICATORI

wind_from_tick = 0
wind_to_tick = 30
wind_range = wind_to_tick - wind_from_tick
x_wind_reg= np.arange(151,302)
x_all_reg=np.arange(302)


#%% UNEMPLOYEMENT RATE (count workers with [not employed?] / count workers)

#Indicatori intera curva
unemp_rate = column_splitter(data, 1000, par="")
unemp_rate_mean = unemp_rate.mean(axis=1)
unemp_rate_median = unemp_rate.median(axis=1)
unemp_rate_max = unemp_rate.max(axis=1)
unemp_rate_min = unemp_rate.min(axis=1)
unemp_rate_std = error_window(unemp_rate.std(axis=1))

#Indicatori Window
unemp_rate_wind = wind_selector(unemp_rate,wind_from_tick,wind_to_tick).mean(axis=1)
unemp_rate_wind_mean = round(unemp_rate_wind.mean(), 6)
unemp_rate_wind_median = round(unemp_rate_wind.median(), 6)
unemp_rate_wind_std = round(unemp_rate_wind.std(),6)
unemp_rate_wind_min = round(unemp_rate_wind.min(), 6)
unemp_rate_wind_max = round(unemp_rate_wind.max(), 6)

#Retta di regressione
unemp_rate_slope, unemp_rate_intercept, unemp_rate_r_value, unemp_rate_p_value, unemp_rate_std_err = stats.linregress(x_wind_reg, unemp_rate_mean[150:302])
unemp_rate_reg_line = unemp_rate_slope * x_all_reg + unemp_rate_intercept 



#%% NOMINAL GDP (ln-hopital nominal-GDP)

#Indicatori intera curva
nominal_GDP = column_splitter(data, 1000, par=".1")
nominal_GDP_mean = nominal_GDP.mean(axis=1)
nominal_GDP_median = nominal_GDP.median(axis=1)
nominal_GDP_max = nominal_GDP.max(axis=1)
nominal_GDP_min = nominal_GDP.min(axis=1)
nominal_GDP_std = error_window(nominal_GDP.std(axis=1))

#Indicatori Window
nominal_GDP_wind = wind_selector(nominal_GDP,wind_from_tick,wind_to_tick).mean(axis=1)
nominal_GDP_wind_mean = round(nominal_GDP_wind.mean(), 6)
nominal_GDP_wind_median = round(nominal_GDP_wind.median(), 6)
nominal_GDP_wind_std = round(nominal_GDP_wind.std(),6)
nominal_GDP_wind_min = round(nominal_GDP_wind.min(), 6)
nominal_GDP_wind_max = round(nominal_GDP_wind.max(), 6)

#Retta di regressione
nominal_GDP_slope, nominal_GDP_intercept, nominal_GDP_r_value, nominal_GDP_p_value, nominal_GDP_std_err = stats.linregress(x_wind_reg, nominal_GDP_mean[150:302])
nominal_GDP_reg_line = nominal_GDP_slope * x_all_reg + nominal_GDP_intercept 


#%% PRODUCTION OF FIRMS (mean [production-Y] of firms

#Indicatori intera curva
production_firm = column_splitter(data, 1000, par=".7")
production_firm_mean = production_firm.mean(axis=1)
production_firm_median = production_firm.median(axis=1)
production_firm_max_mean = column_splitter(data, 1000, par=".6").mean(axis=1)
production_firm_min_mean = column_splitter(data, 1000, par=".8").mean(axis=1)
production_firm_std = error_window(production_firm.std(axis=1))

#Indicatori Window
production_firm_wind = wind_selector(production_firm,wind_from_tick,wind_to_tick).mean(axis=1)
production_firm_wind_mean = round(production_firm_wind.mean(), 6)
production_firm_wind_median = round(production_firm_wind.median(), 6)
production_firm_wind_std = round(production_firm_wind.std(),6)
production_firm_wind_min = wind_selector(production_firm_min_mean, wind_from_tick, wind_to_tick).mean()
production_firm_wind_max = wind_selector(production_firm_max_mean, wind_from_tick, wind_to_tick).mean()

#Retta di regressione
production_firm_slope, production_firm_intercept, production_firm_r_value, production_firm_p_value, production_firm_std_err = stats.linregress(x_wind_reg, production_firm_mean[150:302])
production_firm_reg_line = production_firm_slope * x_all_reg + production_firm_intercept 


#%% WAGE OFFERED (mean [wage-offered-Wb] of firms)

#Indicatori intera curva
wage = column_splitter(data, 1000, par=".9")
wage_mean = wage.mean(axis=1)
wage_median = wage.median(axis=1)
wage_std = error_window(wage.std(axis=1))
wage_max_mean =column_splitter(data, 1000, par=".11").mean(axis=1)
wage_min_mean = column_splitter(data, 1000, par=".10").mean(axis=1)

#Indicatori Window
wage_wind = wind_selector(wage,wind_from_tick,wind_to_tick).mean(axis=1)
wage_wind_mean = round(production_firm_wind.mean(), 6)
wage_wind_median = round(production_firm_wind.median(), 6)
wage_wind_std = round(production_firm_wind.std(),6)
wage_wind_min = wind_selector(wage_min_mean, wind_from_tick, wind_to_tick).mean()
wage_wind_max = wind_selector(wage_max_mean, wind_from_tick, wind_to_tick).mean()

#Retta di regressione
wage_slope, wage_intercept, wage_r_value, wage_p_value, wage_std_err = stats.linregress(x_wind_reg, wage_mean[150:302])
wage_reg_line = wage_slope * x_all_reg + wage_intercept 



#%% WEALTH (ln-hopital mean [wealth] of workers)

#Indicatori intera curva
wealth = column_splitter(data, 1000, par=".12")
wealth_mean = wealth.mean(axis=1)
wealth_median = wealth.median(axis=1)
wealth_std = error_window(wealth.std(axis=1))
wealth_max_mean = column_splitter(data, 1000, par=".13").mean(axis=1)
wealth_min_mean = column_splitter(data, 1000, par=".14").mean(axis=1)

#Indicatori Window
wealth_wind = wind_selector(wealth,wind_from_tick,wind_to_tick).mean(axis=1)
wealth_wind_mean = round(wealth_wind.mean(), 6)
wealth_wind_median = round(wealth_wind.median(), 6)
wealth_wind_std = round(wealth_wind.std(),6)
wealth_wind_min = wind_selector(wealth_mean, wind_from_tick, wind_to_tick).mean()
wealth_wind_max = wind_selector(wealth_mean, wind_from_tick, wind_to_tick).mean()

#Retta di regressione
wealth_slope, wealth_intercept, wealth_r_value, wealth_p_value, wealth_std_err = stats.linregress(x_wind_reg, wealth_mean[150:302])
wealth_reg_line = wealth_slope * x_all_reg + wealth_intercept 


#%% INTEREST RATE (mean [my-interest-rate] of firms)

#Indicatori intera curva
interest_rate = column_splitter(data, 1000, par=".18")
interest_rate_mean = interest_rate.mean(axis=1)
interest_rate_median = interest_rate.median(axis=1)
interest_rate_std = error_window(interest_rate.std(axis=1))
interest_rate_max_mean = column_splitter(data, 1000, par=".19").mean(axis=1)
interest_rate_min_mean = column_splitter(data, 1000, par=".20").mean(axis=1)

#Indicatori Window
interest_rate_wind = wind_selector(interest_rate,wind_from_tick,wind_to_tick).mean(axis=1)
interest_rate_wind_mean = round(interest_rate_wind.mean(), 6)
interest_rate_wind_median = round(interest_rate_wind.median(), 6)
interest_rate_wind_std = round(interest_rate_wind.std(),6)
interest_rate_wind_min = wind_selector(interest_rate_mean, wind_from_tick, wind_to_tick).mean()
interest_rate_wind_max = wind_selector(interest_rate_mean, wind_from_tick, wind_to_tick).mean()

#Retta di regressione
interest_rate_slope, interest_rate_intercept, interest_rate_r_value, interest_rate_p_value, interest_rate_std_err = stats.linregress(x_wind_reg, interest_rate_mean[150:302])
interest_rate_reg_line = interest_rate_slope * x_all_reg + interest_rate_intercept 



#%% GRAFICO
x= np.arange(0,301)
fig = make_subplots(rows=6, cols=2, 
                    vertical_spacing=0.019,
                    horizontal_spacing=0.019,
                    specs=[[{"type": "scatter"}, {"type": "table"}],
                           [{"type": "scatter"}, {"type": "table"}],
                           [{"type": "scatter"}, {"type": "table"}],
                           [{"type": "scatter"}, {"type": "table"}],
                           [{"type": "scatter"}, {"type": "table"}],
                           [{"type": "scatter"}, {"type": "table"}]], 
                    x_title="tick", 
                    column_widths=[0.8, 0.2], 
                    subplot_titles=("Unemployment rate","Indicators (tick)",
                                    "Nominal GDP", "Indicators (tick)",
                                    "Production of firms","Indicators (tick)",
                                    "Wage Offered","Indicators (tick)",
                                    "Wealth of workers","Indicators (tick)",
                                    "Contractual interest rate","Indicators (tick)"))

#%% Unemployment rate
fig.add_trace(go.Scatter(x=x,y=unemp_rate_max,name="max",line=dict(width=0.8)), row=1, col=1)
fig.add_trace(go.Scatter(x=x,y=unemp_rate_median,name="median", line=dict(width=1.0), error_y=dict(type='data', array=unemp_rate_std, symmetric=True)),row=1, col=1)
fig.add_trace(go.Scatter(x=x,y=unemp_rate_min,name="min",line=dict(width=0.8)), row=1, col=1)
fig.add_trace(go.Scatter(x=x,y=unemp_rate_reg_line,name="reg_line",line=dict(width=0.5)), row=1, col=1)
                           
fig.add_trace(go.Table(header=dict(values=['Indicator', 'Score'], align=['left', 'center']), 
                       cells=dict(values=[["Mean (280-300)", 
                                           "Median (280-300)",
                                           "Min (280-300)", 
                                           "Max (280-300)", 
                                           "Std (280-300)", 
                                           "Reg-line Slope (150-300)", 
                                           "Reg-line Intercept (150-300)",
                                           "Reg-line r-value (150-300)",
                                           "Reg-line p-value (150-300)",
                                           "Reg-line std error (150-300)"], [unemp_rate_wind_mean, 
                                                                            unemp_rate_wind_median,
                                                                            unemp_rate_wind_min,
                                                                            unemp_rate_wind_max,
                                                                            unemp_rate_wind_std,
                                                                            round(unemp_rate_slope,6),
                                                                            round(unemp_rate_intercept,6),
                                                                            round(unemp_rate_r_value,6),
                                                                            round(unemp_rate_p_value,6),
                                                                            round(unemp_rate_std_err,6)]], align=['left', 'left'])), row=1, col=2)


#%% Nominal GDP
fig.add_trace(go.Scatter(x=x,y=nominal_GDP_max,name="max",line=dict(width=0.8)), row=2, col=1)
fig.add_trace(go.Scatter(x=x,y=nominal_GDP_median,name="median", line=dict(width=1.0), error_y=dict(type='data', array=nominal_GDP_std, symmetric=True)),row=2, col=1)
fig.add_trace(go.Scatter(x=x,y=nominal_GDP_min,name="min",line=dict(width=0.8)), row=2, col=1)
fig.add_trace(go.Scatter(x=x,y=nominal_GDP_reg_line,name="reg_line",line=dict(width=0.5)), row=2, col=1)
                           
fig.add_trace(go.Table(header=dict(values=['Indicator', 'Score'], align=['left', 'center']), 
                       cells=dict(values=[["Mean (280-300)", 
                                           "Median (280-300)",
                                           "Min (280-300)", 
                                           "Max (280-300)", 
                                           "Std (280-300)", 
                                           "Reg-line Slope (150-300)", 
                                           "Reg-line Intercept (150-300)",
                                           "Reg-line r-value (150-300)",
                                           "Reg-line p-value (150-300)",
                                           "Reg-line std error (150-300)"], [nominal_GDP_wind_mean, 
                                                                            nominal_GDP_wind_median,
                                                                            nominal_GDP_wind_min,
                                                                            nominal_GDP_wind_max,
                                                                            nominal_GDP_wind_std,
                                                                            round(nominal_GDP_slope,6),
                                                                            round(nominal_GDP_intercept,6),
                                                                            round(nominal_GDP_r_value,6),
                                                                            round(nominal_GDP_p_value,6),
                                                                            round(nominal_GDP_std_err,6)]], align=['left', 'left'])), row=2, col=2)


#%% Production of firms
fig.add_trace(go.Scatter(x=x,y=production_firm_max_mean,name="max",line=dict(width=0.8)), row=3, col=1)
fig.add_trace(go.Scatter(x=x,y=production_firm_median,name="median", line=dict(width=1.0), error_y=dict(type='data', array=production_firm_std, symmetric=True)),row=3, col=1)
fig.add_trace(go.Scatter(x=x,y=production_firm_min_mean,name="min",line=dict(width=0.8)), row=3, col=1)
fig.add_trace(go.Scatter(x=x,y=production_firm_reg_line,name="reg_line",line=dict(width=0.5)), row=3, col=1)
                           
fig.add_trace(go.Table(header=dict(values=['Indicator', 'Score'], align=['left', 'center']), 
                       cells=dict(values=[["Mean (280-300)", 
                                           "Median (280-300)",
                                           "Min (280-300)", 
                                           "Max (280-300)", 
                                           "Std (280-300)", 
                                           "Reg-line Slope (150-300)", 
                                           "Reg-line Intercept (150-300)",
                                           "Reg-line r-value (150-300)",
                                           "Reg-line p-value (150-300)",
                                           "Reg-line std error (150-300)"], [production_firm_wind_mean, 
                                                                            production_firm_wind_median,
                                                                            round(production_firm_wind_min,6),
                                                                            round(production_firm_wind_max, 6),
                                                                            production_firm_wind_std,
                                                                            round(production_firm_slope,6),
                                                                            round(production_firm_intercept,6),
                                                                            round(production_firm_r_value,6),
                                                                            round(production_firm_p_value,6),
                                                                            round(production_firm_std_err,6)]], align=['left', 'left'])), row=3, col=2)
                                                                                                                 
                                                                            


#%% Wage Offered
fig.add_trace(go.Scatter(x=x,y=wage_max_mean,name="max",line=dict(width=0.8)), row=4, col=1)
fig.add_trace(go.Scatter(x=x,y=wage_median,name="median", line=dict(width=1.0), error_y=dict(type='data', array=wage_std, symmetric=True)),row=4, col=1)
fig.add_trace(go.Scatter(x=x,y=wage_min_mean,name="min",line=dict(width=0.8)), row=4, col=1)
fig.add_trace(go.Scatter(x=x,y=wage_reg_line,name="reg_line",line=dict(width=0.5)), row=4, col=1)
                           
fig.add_trace(go.Table(header=dict(values=['Indicator', 'Score'], align=['left', 'center']), 
                       cells=dict(values=[["Mean (280-300)", 
                                           "Median (280-300)",
                                           "Min (280-300)", 
                                           "Max (280-300)", 
                                           "Std (280-300)", 
                                           "Reg-line Slope (150-300)", 
                                           "Reg-line Intercept (150-300)",
                                           "Reg-line r-value (150-300)",
                                           "Reg-line p-value (150-300)",
                                           "Reg-line std error (150-300)"], [wage_wind_mean, 
                                                                            wage_wind_median,
                                                                            round(wage_wind_min, 6),
                                                                            round(wage_wind_max, 6),
                                                                            wage_wind_std,
                                                                            round(wage_slope,6),
                                                                            round(wage_intercept,6),
                                                                            round(wage_r_value,6),
                                                                            round(wage_p_value,6),
                                                                            round(wage_std_err,6)]], align=['left', 'left'])), row=4, col=2)

#%% Wealth of workers
fig.add_trace(go.Scatter(x=x,y=wealth_max_mean,name="max",line=dict(width=0.8)), row=5, col=1)
fig.add_trace(go.Scatter(x=x,y=wealth_median,name="median", line=dict(width=1.0), error_y=dict(type='data', array=wealth_std, symmetric=True)),row=5, col=1)
fig.add_trace(go.Scatter(x=x,y=wealth_min_mean,name="min",line=dict(width=0.8)), row=5, col=1)
fig.add_trace(go.Scatter(x=x,y=wealth_reg_line,name="reg_line",line=dict(width=0.5)), row=5, col=1)
                           
fig.add_trace(go.Table(header=dict(values=['Indicator', 'Score'], align=['left', 'center']), 
                       cells=dict(values=[["Mean (280-300)", 
                                           "Median (280-300)",
                                           "Min (280-300)", 
                                           "Max (280-300)", 
                                           "Std (280-300)", 
                                           "Reg-line Slope (150-300)", 
                                           "Reg-line Intercept (150-300)",
                                           "Reg-line r-value (150-300)",
                                           "Reg-line p-value (150-300)",
                                           "Reg-line std error (150-300)"], [wealth_wind_mean, 
                                                                            wealth_wind_median,
                                                                            round(wealth_wind_min,6),
                                                                            round(wealth_wind_max, 6),
                                                                            wealth_wind_std,
                                                                            round(wealth_slope,6),
                                                                            round(wealth_intercept,6),
                                                                            round(wealth_r_value,6),
                                                                            round(wealth_p_value,6),
                                                                            round(wealth_std_err,6)]], align=['left', 'left'])), row=5, col=2)

#%% Contractual interest rate
fig.add_trace(go.Scatter(x=x,y=interest_rate_max_mean,name="max",line=dict(width=0.8)), row=6, col=1)
fig.add_trace(go.Scatter(x=x,y=interest_rate_median,name="median", line=dict(width=1.0), error_y=dict(type='data', array=interest_rate_std, symmetric=True)),row=6, col=1)
fig.add_trace(go.Scatter(x=x,y=interest_rate_min_mean,name="min",line=dict(width=0.8)), row=6, col=1)
fig.add_trace(go.Scatter(x=x,y=interest_rate_reg_line,name="reg_line",line=dict(width=0.5)), row=6, col=1)
                           
fig.add_trace(go.Table(header=dict(values=['Indicator', 'Score'], align=['left', 'center']), 
                       cells=dict(values=[["Mean (280-300)", 
                                           "Median (280-300)",
                                           "Min (280-300)", 
                                           "Max (280-300)", 
                                           "Std (280-300)", 
                                           "Reg-line Slope (150-300)", 
                                           "Reg-line Intercept (150-300)",
                                           "Reg-line r-value (150-300)",
                                           "Reg-line p-value (150-300)",
                                           "Reg-line std error (150-300)"], [interest_rate_wind_mean, 
                                                                            interest_rate_wind_median,
                                                                            round(interest_rate_wind_min,6),
                                                                            round(interest_rate_wind_max, 6),
                                                                            interest_rate_wind_std,
                                                                            round(interest_rate_slope,6),
                                                                            round(interest_rate_intercept,6),
                                                                            round(interest_rate_r_value,6),
                                                                            round(interest_rate_p_value,6),
                                                                            round(interest_rate_std_err,6)]], align=['left', 'left'])), row=6, col=2)






fig.update_layout(height=2600, width=1600, showlegend=False)
plot(fig)





















