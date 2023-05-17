#!/usr/bin/python3
from myTools import *
pd.set_option('display.max_columns', 500)
input = pd.read_csv("Forecast_13_05_2023.csv")
# input = input.iloc[50:]
input = input[["t2m", "u10", "v10", "tp", "time"]].copy()
input.t2m = input.t2m -273.15
res = calcWind(input["u10"], input["v10"])
input["wind_direction"] = res[1].tolist()
input["wind_speed"] = res[0].tolist()
input.drop(columns=["u10", "v10"], inplace=True)
input = input[["t2m", "wind_direction", "wind_speed", "tp", "time"]]
input = matchTraining(input)# predictions = treeRegressor(input)

# scaling input data
scaler = pickle.load(open("inputScaler", "rb"))
input[input.columns] = scaler.transform(input)

predictions = forestRegressor(input)
quantiles = quantiles(predictions)
quantiles.to_csv("20230513_JustinBieber.csv")
print("done")