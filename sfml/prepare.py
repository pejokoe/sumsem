#!/usr/bin/python3

import pandas as pd
import numpy as np

apartments = pd.read_csv("immo_data.csv")
apartments = apartments[["serviceCharge", "picturecount" ,"pricetrend", "telekomUploadSpeed", "totalRent", "yearConstructed", "baseRent", "livingSpace", "garden"]]
apartments = apartments.dropna()
print(apartments.head())
apartments.to_csv("/home/swt/Programme/sumsem/sfml/immo_cleaned.csv")
print("done")