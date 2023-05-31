#!/usr/bin/python3
from myTools import *
pd.set_option('display.max_columns', 500)

input = pd.read_csv("Forecast_06_05_2023.csv")

# use this line if 0th ensemble is present
# input = input.iloc[50:]

# matching the input data to the training data format
input = input[["t2m", "u10", "v10", "tp", "time"]].copy()
input.t2m = input.t2m -273.15
res = calcWind(input["u10"], input["v10"])
input["wind_direction"] = res[1].tolist()
input["wind_speed"] = res[0].tolist()
input.drop(columns=["u10", "v10"], inplace=True)
input = input[["t2m", "wind_direction", "wind_speed", "tp", "time"]]

# scaling input data
scaler = pickle.load(open("inputScaler", "rb"))
input[input.columns] = scaler.transform(input)

# make predictions using random forests and calculate quantiles
predictions = forestRegressor(input)
quantiles = quantiles(predictions)

# save result as latex table for report
with open("quantiles.txt", "w") as file:
    file.write(quantiles[:5].to_latex(float_format="%.3f", longtable=True))
quantiles.to_csv("20230613_JustinBieber.csv")
print("done")