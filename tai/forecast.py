#!/usr/bin/python3
from myTools import *


data = pd.read_csv("Forecast_01_04_2023.csv")

data["wind"] =  calcWind(data["u10"], data["v10"])
data["rh"] = calcRh(data["d2m"], data["t2m"])



data["forecast_date"] = data["time"].str.split(" ").str[0]
data["horizon"] = data["time"].str.split(" ").str[1].str[0] + " hour"
data["precip"] = data["tp"]
dataToProcess = data[["time", "t2m", "wind", "precip", "rh"]].copy()

newData = {"forecast_date":[], "target":[], "horizon":[], "q0.025":[], "q0.25":[], "q0.5":[],
                         "q0.75":[], "q0.975":[]}

labels = ["wind", "t2m", "rh", "precip"]
timePoints = pd.unique(data["time"])
time = str(timePoints[0].split(" ")[0])
hours = 0
skipFirst = 0
for timePoint in timePoints:
    if skipFirst:
        current = data.loc[data["time"] == timePoint]
        for label in labels:
            newData["forecast_date"].append(time)
            newData["target"].append(label)
            newData["horizon"].append(str(hours) + " hour")
            newData["q0.025"].append(np.quantile(current[label], 0.025))
            newData["q0.25"].append(np.quantile(current[label], 0.25))
            newData["q0.5"].append(np.quantile(current[label], 0.5))
            newData["q0.75"].append(np.quantile(current[label], 0.75))
            newData["q0.975"].append(np.quantile(current[label], 0.975))
    hours += 6
    skipFirst = 1

newDataFrame = pd.DataFrame.from_dict(newData)
newDataFrame.to_csv("20230401_JustinBieber.csv")
        

# quantiles = dict.fromkeys("forecast_date", "target", "horizon", "q0.025", 
#                           "q0.25", "q0.5", "q0.75", "q0.975")
# print(data["horizon"])
# for label in labels:
