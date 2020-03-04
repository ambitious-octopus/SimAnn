# -*- coding: utf-8 -*-
"""
Created by Francesco Mattioli
Francesco@nientepanico.org
www.nientepanico.org
"""

#SCARICARE DATABASE QUI https://1drv.ms/u/s!Ai6q7IUrZTevhZ9VbOjqMoVZ6b3eWA?e=cvlRRG

import pandas as pd
import numpy as np

def reader(file):
    data = pd.read_csv(file, skiprows=6)
    return data

raw_data = reader("run#2-spreadsheet.csv")