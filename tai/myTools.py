import numpy as np
import pandas as pd

def calcWind(u, v):
    return np.array((u * u + v * v).pow(0.5))

def calcRh(d, t):
    return np.array(100*(np.exp((d * 17.625) / (234.04 + d)) / 
                           (np.exp((t * 17.625) / (234.04  + t)))))

def formatDate(dataframe):
    dataframe["forecast_date"] = str(dataframe["time"]).split(" ")[0]
    dataframe["horizon"] = str(dataframe["time"]).split(" ")[1].split(":")[0]  + "hour"