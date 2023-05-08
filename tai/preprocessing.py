#!/usr/bin/python3
from myTools import *
from ast import literal_eval

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
allData.tp6 = allData.tp6 * 1000

# normalize data using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
print(allData.head())
allData[["t2m", "wind_direction", "wind_speed", "tp6"]] = scaler.fit_transform(allData[["t2m", "wind_direction", "wind_speed", "tp6"]])
print(allData.head())

allData = splitInTwenty(allData)

#target data
syn2017 = pd.read_csv("synop_2017_March_June.csv")
syn2018 = pd.read_csv("synop_2018_March_June.csv")
synop = pd.concat([syn2017, syn2018], axis = 0)
synop = synop.iloc[1::2]
synop["precip_quantity_6hr"] = np.append(np.array([0]), accumulateTp(synop["precip_quantity_1hour"]))[:-1]
synop = synop.drop(columns=["humidity_relative", "precip_quantity_1hour", "local_datetime", "name", "lat", "lon", "community_name"])
synop = synop[synop["datetime"].str.contains("00:00:00") | synop["datetime"].str.contains("06:00:00") |
                  synop["datetime"].str.contains("12:00:00") | synop["datetime"].str.contains("18:00:00")]
synop = synop.reset_index(drop=True)
# trainingSet = createTrainingSetTimeSeries(allData, synop)
trainingSet = createTrainingSet(allData, synop)
trainingSet.to_csv("TrainingSet.csv")

# number format from string format
trainingSet = pd.read_csv("TrainingSet.csv")
trainingSet.dropna(inplace=True)
trainingSet = trainingSet[trainingSet.Target.str.contains("nan")==False]
trainingSet = trainingSet.reset_index()
input = trainingSet.Input.apply(literal_eval)
target = trainingSet.Target.apply(literal_eval)
input = np.array(input.tolist())
target = np.array(target.tolist())
np.savetxt("Input.csv", input, delimiter=",")
np.savetxt("Target.csv", target, delimiter=",")
# print(type(res[0]))
# print(len(res[0]))
# print(surface.head(10))