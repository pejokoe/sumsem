#!/usr/bin/python3
from myTools import *
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle

# read training files
input = np.genfromtxt("InputTimeSeries.csv", delimiter=",")
target = np.genfromtxt("TargetTimeSeries.csv", delimiter=",")

# extract distinct test set, not to be touched
xTrain, xTest, yTrain, yTest = train_test_split(input, target, test_size=0.1, random_state = 0, shuffle=True)

start, end = 5, 15
# trees(xTrain, yTrain)
randomForest(xTrain, yTrain)
