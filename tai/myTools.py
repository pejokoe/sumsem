import numpy as np
import pandas as pd
import math 

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
        frames.append(surface[i*20:i*21+20:1])
    return frames

def accumulateTp(precip):
    return precip.rolling(6).sum()

def createTrainingSet(input, target):
    oneTrainingInput=[]
    oneTarget=[]
    factor = 0
    for frame in input:
        for i in range(len(frame)-1):
            if i > 0:
                oneTrainingInput.append(list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i-1])
                + list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i])
                + list(frame[["t2m", "wind_direction", "wind_speed", "tp6"]].iloc[i+1]))
                time = frame.valid_time.iloc[i]
                correctTarget = target[target.local_datetime == time.replace("/", "-")][["temp", "wind_direction", "wind_speed", "precip_quantity_6hr"]]
                oneTarget.append(list(correctTarget.iloc[0]))
    trainingPoints = list(zip(oneTrainingInput, oneTarget))
    print(trainingPoints[:10])
