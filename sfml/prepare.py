#!/usr/bin/python3

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import time

hospital = pd.read_csv("hospital.csv")
hospital = hospital.select_dtypes(include=np.number)
remove_columns = [col for col in hospital.columns if "pache" in col]
hospital = hospital.drop(columns=["Unnamed: 83"])
hospital = hospital.drop(columns=remove_columns)
print(hospital.head())
# x_new = SelectKBest(f_classif, k=3).fit_transform(hospital.drop(columns=["hospital_death"]), hospital.hospital_death)
# print(x_new[:10])
hospital.to_csv("hosp_cleaned.csv")
print("done")