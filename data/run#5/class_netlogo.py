# -*- coding: utf-8 -*-
"""
Created by Francesco Mattioli
Francesco@nientepanico.org
www.nientepanico.org
"""

import numpy as np
import pandas as pd

class NetLogo():
    def reader(self, file):
        data = pd.read_csv(file, skiprows=6)
        return data
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
        for a in range(1,num_run):
            lista.append(str(a)+par)
        colonne = data[lista]
        return colonne



    
        
        