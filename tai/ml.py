#!/usr/bin/python3

import pandas as pd
import numpy as np
import math
from myTools import calcWind, calcRh, formatDate


data = pd.read_csv("Forecast_01_04_2023.csv")
print(data.head)