#!/usr/bin/python3

import pandas as pd
import numpy as np
import math
from myTools import *
import sys

pd.set_option('display.max_columns', 500)


precip = pd.read_csv("ECMWF_2017_2018_precip (1).csv")
precip = precip[precip.number != 0]
surface = pd.read_csv("ECMWF_2017_2018_surface.csv")
surface = surface[surface.number != 0]
surface = surface[surface.step != "0 days 00:00:00"]
syn2017 = pd.read_csv("synop_2017_March_June.csv")
syn2017 = syn2017.iloc[1::2]
res = calcWind(surface["u10"], surface["v10"])


surface["wind_direction"] = res[1].tolist()
surface["wind_speed"] = res[0].tolist()
surface["tp6"] = precip["tp6"]
allData = surface.drop(columns=["time", "step", "surface", "depthBelowLandLayer", "cape", "cin", "sd", "stl1", "swvl1", "tcc", "tcw", "tcwv", "u10", "u100", "v10", "v100", "vis", "model_altitude", "model_land_usage", 
              "model_latitude", "model_longitude", "model_orography", "valid_time"])

allData = allData.reset_index(drop=True)
print(allData["tp6"].loc[1])
# listOfFrames = splitInTwenty(surface)
# trainingSet = createTrainingSet(listOfFrames)
# surInterpolated = interpolate(listOfFrames)
# print(surInterpolated[-100:])
# surInterpolated.to_csv("surInterpolated.csv")
# print(type(res[0]))
# print(len(res[0]))
# print(surface.head(10))