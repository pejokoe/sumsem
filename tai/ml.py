#!/usr/bin/python3

import pandas as pd
import numpy as np
import math
from myTools import *
import sys

pd.set_option('display.max_columns', 500)

#training data
precip = pd.read_csv("ECMWF_2017_2018_precip (1).csv")
precip = precip[precip.number != 0]
surface = pd.read_csv("ECMWF_2017_2018_surface.csv")
surface = surface[surface.number != 0]
surface = surface[surface.step != "0 days 00:00:00"]
res = calcWind(surface["u10"], surface["v10"])

precip.reset_index(drop=True)
surface.reset_index(drop=True)
surface["wind_direction"] = res[1].tolist()
surface["wind_speed"] = res[0].tolist()
surface["tp6"] = precip["tp6"].tolist()
allData = surface.drop(columns=["time", "step", "surface", "depthBelowLandLayer", "cape", "cin", "sd", "stl1", "swvl1", "tcc", "tcw", "tcwv", "u10", "u100", "v10", "v100", "vis", "model_altitude", "model_land_usage", 
              "model_latitude", "model_longitude", "model_orography"])

allData = allData.reset_index(drop=True)
allData.t2m = allData.t2m -273.15
print(allData[:10])
allData = splitInTwenty(surface)

#target data
syn2017 = pd.read_csv("synop_2017_March_June.csv")
syn2017 = syn2017.iloc[1::2]
syn2017 = syn2017.reset_index(drop=True)
syn2017["precip_quantity_6hr"] = accumulateTp(syn2017["precip_quantity_1hour"]).tolist()
syn2017 = syn2017.drop(columns=["humidity_relative", "precip_quantity_1hour", "datetime", "name", "lat", "lon", "community_name"])
syn2017 = syn2017[syn2017["local_datetime"].str.contains("00:00:00") | syn2017["local_datetime"].str.contains("06:00:00") |
                  syn2017["local_datetime"].str.contains("12:00:00") | syn2017["local_datetime"].str.contains("18:00:00")]
# syn2017 = syn2017[ syn2017.local_datetime]
# trainingSet = createTrainingSet(allData)
# surInterpolated = interpolate(listOfFrames)
# print(surInterpolated[-100:])
# surInterpolated.to_csv("surInterpolated.csv")
# print(type(res[0]))
# print(len(res[0]))
# print(surface.head(10))