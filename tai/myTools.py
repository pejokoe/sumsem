#!/usr/bin/python3
import numpy as np
import pandas as pd
import math 
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt


def calcAngle(o_leg, a_leg):
    if o_leg > 0 and a_leg > 0:
        o_leg = abs(o_leg)
        a_leg = abs(a_leg)
        return math.degrees(math.atan(o_leg / a_leg)) + 180

    if o_leg < 0 and a_leg < 0:
        o_leg = abs(o_leg)
        a_leg = abs(a_leg)
        return math.degrees(math.atan(o_leg / a_leg))
    
    if o_leg < 0 and a_leg > 0:
        o_leg = abs(o_leg)
        a_leg = abs(a_leg)
        return 180 - math.degrees(math.atan(o_leg / a_leg))
    
    if o_leg > 0 and a_leg < 0:
        o_leg = abs(o_leg)
        a_leg = abs(a_leg)
        return 360 - math.degrees(math.atan(o_leg / a_leg))

def calcWind(u, v):
    speed = np.array((u * u + v * v).pow(0.5))
    helper = np.vectorize(calcAngle)
    direction = helper(u, v)
    return np.array([speed, direction])

def calcRh(d, t):
    return np.array(100*(np.exp((d * 17.625) / (234.04 + d)) / 
                           (np.exp((t * 17.625) / (234.04  + t)))))

def formatDate(dataframe):
    dataframe["forecast_date"] = str(dataframe["time"]).split(" ")[0]
    dataframe["horizon"] = str(dataframe["time"]).split(" ")[1].split(":")[0]  + "hour"

def interpolate(frames):
    'linearly interpolate values for every full hour'
    newFrames = []
    for ele in frames:
        insert = 6
        ele.index = range(0, (len(ele)) * insert, insert)
        ele = ele.reindex(index=range((len(ele)-1)*insert + 1))
        ele = ele.interpolate()
        newFrames.append(ele)
    surInterpolated = pd.concat(newFrames, ignore_index=True)
    print(surInterpolated.shape)
    return surInterpolated

def splitInTwenty(surface):

    frames = []
    for i in range(int(len(surface)/20)):
        frames.append(surface[i*20:i*20+20:1])
    return frames

def accumulateTp(precip):
    accumulated = np.array(precip.rolling(6).sum())
    return accumulated

def createTrainingSet(input, target):
    oneTrainingInput=[]
    oneTarget=[]

    for frame in input:
        for i in range(len(frame)):
            if i > 0 and i < len(frame) - 1:
                oneTrainingInput.append(list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1])
                + list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i])
                + list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1]))
            elif i == 0:
                oneTrainingInput.append(list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i])
                + list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i])
                + list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1]))
            else:
                oneTrainingInput.append(list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1])
                + list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i])
                + list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i]))
            time = frame.valid_time.iloc[i]
            correctTarget = target[target.datetime == time][["temp", "wind_direction", "wind_speed", "precip_quantity_6hr"]]
            oneTarget.append(list(correctTarget.iloc[0]))
    return pd.DataFrame(list(zip(oneTrainingInput, oneTarget)), columns=["Input", "Target"])

def matchTraining(input):
    sets = []
    oneInput = []
    for i in range(50):
        sets.append(input[i::50])
    for set in sets:
        for i in range(len(set)):
            if i > 0 and i < len(set) - 1:
                oneInput.append(list(set[["t2m", "wind_direction", "wind_speed", "tp"]].iloc[i-1])
                + list(set[["t2m", "wind_direction", "wind_speed", "tp"]].iloc[i])
                + list(set[["t2m", "wind_direction", "wind_speed", "tp"]].iloc[i+1]))
            elif i == 0:
                oneInput.append(list(set[["t2m", "wind_direction", "wind_speed", "tp"]].iloc[i])
                + list(set[["t2m", "wind_direction", "wind_speed", "tp"]].iloc[i])
                + list(set[["t2m", "wind_direction", "wind_speed", "tp"]].iloc[i+1]))
            else:
                oneInput.append(list(set[["t2m", "wind_direction", "wind_speed", "tp"]].iloc[i-1])
                + list(set[["t2m", "wind_direction", "wind_speed", "tp"]].iloc[i])
                + list(set[["t2m", "wind_direction", "wind_speed", "tp"]].iloc[i]))
    return np.array(oneInput)

def treeRegressor(input):
    temperatureTree = pickle.load(open("TemperatureTree", "rb"))
    windTree = pickle.load(open("WindTree", "rb"))
    precipTree = pickle.load(open("PrecipTree", "rb"))
    predTemp = temperatureTree.predict(input)
    predWind = windTree.predict(input)
    predPrecip = precipTree.predict(input)
    predictions = pd.DataFrame(columns=["t2m", "wind", "precip"])
    predictions["t2m"] = predTemp
    predictions["wind"] = predWind
    predictions["precip"] = predPrecip
    return predictions

def quantiles(predictions):
    result = pd.DataFrame(columns=["forecast_date", "target", "horizon", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"])
    quants = [0.025, 0.25, 0.5, 0.75, 0.975]
    for i in range(0, 20):
        horizon = i * 6 + 6
        quantsTimestepTemp = []
        quantsTimestepWind = []
        quantsTimestepPrecip = []
        for quant in quants:
            quantsTimestepTemp.append(np.quantile(predictions["t2m"][i::20], quant))
            quantsTimestepWind.append(np.quantile(predictions["wind"][i::20], quant))
            quantsTimestepPrecip.append(np.quantile(predictions["precip"][i::20], quant))
        result = result.append(pd.Series(["2023-04-29", "t2m", str(horizon) + " hour", *quantsTimestepTemp], 
                                         index=["forecast_date", "target", "horizon", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]),
                                         ignore_index=True)
        result = result.append(pd.Series(["2023-04-29", "wind", str(horizon) + " hour", *quantsTimestepWind], 
                               index=["forecast_date", "target", "horizon", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]), 
                               ignore_index=True)
        result = result.append(pd.Series(["2023-04-29", "precip", str(horizon) + " hour", *quantsTimestepPrecip], 
                               index=["forecast_date", "target", "horizon", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]), 
                               ignore_index=True)
    return result

def validation(model, input, target):
    'Function for k-fold cross validation, returns training and validation rmse'
    kf = KFold(5, shuffle = True, random_state = 0)
    rsme = [[], []]
    for train_ind, val_ind in kf.split(input, target):
        model.fit(input[train_ind], target[train_ind])
        rsme[0].append(mean_squared_error(model.predict(input[train_ind]), target[train_ind])**0.5)
        rsme[1].append(mean_squared_error(model.predict(input[val_ind]), target[val_ind])**0.5)
    return [np.mean(rsme[0]), np.mean(rsme[1])]

def trees(parameters, xTrain, yTrain):
    start = parameters[0]
    end = parameters[1]
    temp = []
    wind = []
    precip = []
    for depth in range(start, end):
        temperatureTree = tree.DecisionTreeRegressor(max_depth=depth)
        windTree = tree.DecisionTreeRegressor(max_depth=depth)
        precipTree = tree.DecisionTreeRegressor(max_depth=depth)
        temp.extend(validation(temperatureTree, xTrain, yTrain[:, 0]))
        wind.extend(validation(windTree, xTrain, yTrain[:, 2]))
        precip.extend(validation(precipTree, xTrain, yTrain[:, 3]))
    fig, axis = plt.subplots(1, 3)
    axis[0].plot(range(start, end), temp[::2])
    axis[0].plot(range(start, end), temp[1::2])
    axis[0].set_title("Temperature")
    axis[0].set_xlabel("Depth of tree")
    axis[0].set_ylabel("RMSE")

    axis[1].plot(range(start, end), wind[::2])
    axis[1].plot(range(start, end), wind[1::2])
    axis[1].set_title("Wind speed")
    axis[1].set_xlabel("Depth of tree")

    axis[2].plot(range(start, end), precip[::2])
    axis[2].plot(range(start, end), precip[1::2])
    axis[2].set_title("Precipitation")
    axis[2].set_xlabel("Depth of tree")

    axis[0].grid()
    axis[1].grid()
    axis[2].grid()
    plt.savefig("test.pdf", format="pdf", bbox_inches="tight")
    plt.show()