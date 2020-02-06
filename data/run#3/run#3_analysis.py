# -*- coding: utf-8 -*-
"""
Created by Francesco Mattioli
Francesco@nientepanico.org
www.nientepanico.org
"""

#SCARICARE DATABASE QUI 

import pandas as pd
import numpy as np

def reader(file):
    data = pd.read_csv(file, skiprows=6)
    return data

raw_data = reader("run#3-spreadsheet.csv")

#Divido raw_data in due parti: data_head con info descrittive e data con i dati attuali
data_head, data= raw_data[:19], raw_data[19:]

#Elimino la prima colonna che non indica una mazza
data = data.drop("[run number]", axis=1)

new_index = np.arange(0,301)


