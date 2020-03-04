# -*- coding: utf-8 -*-
"""
Created by Francesco Mattioli
Francesco@nientepanico.org
www.nientepanico.org
"""

import pandas as pd
import numpy as np

def reader(file):
    data = pd.read_csv(file, skiprows=6)
    return data


data = reader("bam_model run#1-spreadsheet.csv")


