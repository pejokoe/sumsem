#!/usr/bin/python3

import pandas as pd
import numpy as np
import math
from myTools import *
import sys



precip = pd.read_csv("ECMWF_2017_2018_precip (1).csv")
surface = pd.read_csv("ECMWF_2017_2018_surface.csv")
syn2017 = pd.read_csv("synop_2017_March_June.csv")
syn2017 = syn2017.iloc[1::2]
# print(surface.head)
res = calcWind(surface["u10"], surface["v10"])

# prepare training examples
myPrecip = pd.DataFrame()
myPrecip["time"] = precip["time"]
myPrecip["step"] = precip["step"]
myPrecip["tp6"] = precip["tp6"]

surface["wind_direction"] = res[1].tolist()
surface["wind_speed"] = res[0].tolist()
surface = surface.drop(columns=["surface", "depthBelowLandLayer", "cape", "cin", "sd", "stl1", "swvl1", "tcc", "tcw", "tcwv", "u100", "v100", "vis", "model_altitude", "model_land_usage", 
              "model_latitude", "model_longitude", "model_orography"])

surInterpolated = interpolate(surface)
surInterpolated.to_csv("surInterpolated.csv")
# print(type(res[0]))
# print(len(res[0]))
# print(surface.head(10))