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

def interpolate(surface):
    'linearly interpolate values for every full hour'
    insert = 5
    surface.index = range(0, len(surface) * insert, insert)
    surInterpolated = surface.reindex(index=range(len(surface)*insert))
    linInterpolated = surInterpolated["wind_direction", "wind_speed","t2m"].interpolate()
    timeInterpolated = surInterpolated["time", "step", "valid_time"].interpolate(method="time")
    surInterpolated = pd.concat(linInterpolated, timeInterpolated)
    print(surInterpolated.head(10))