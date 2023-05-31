#!/usr/bin/python3

import pandas as pd
import numpy as np

data = pd.read_csv("events.csv")
print(len(data))
data = data[data.location == 14]
print(len(data))
# data.to_csv("pen.csv")
# data = data[data.shot_outcome == 1]
# data = data[data.event_type == 1]
# goal = data[data.is_goal == 1]
# no_goal = data[data.is_goal == 0]
# print(len(goal))
# print(len(no_goal))