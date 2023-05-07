#!/usr/bin/python3
from myTools import *
from sklearn import tree
from sklearn.model_selection import train_test_split
from ast import literal_eval
import matplotlib.pyplot as plt
import pickle
trainingSet = pd.read_csv("TrainingSet.csv")
trainingSet.dropna(inplace=True)
trainingSet = trainingSet[trainingSet.Target.str.contains("nan")==False]
trainingSet = trainingSet.reset_index()
input = trainingSet.Input.apply(literal_eval)
target = trainingSet.Target.apply(literal_eval)
input = np.array(input.tolist())
target = np.array(target.tolist())
np.savetxt("Input.csv", input, delimiter=",")
np.savetxt("Target.csv", target, delimiter=",")
input = np.genfromtxt("Input.csv", delimiter=",")
target = np.genfromtxt("Target.csv", delimiter=",")
# xTrain, xTest, yTrain, yTest = train_test_split(input, target, test_size=0.25, random_state = 0, shuffle=True)

# rmse = {"temp":[], "wind":[], "precip":[]}
# tempMin = 100
# windMin = 100
# precipMin = 100
# start, end = 3, 17
# bestTree = [[], [], []]
# for depth in range(start, end):
#     temperatureTree = tree.DecisionTreeRegressor(max_depth=depth)
#     windTree = tree.DecisionTreeRegressor(max_depth=depth)
#     precipTree = tree.DecisionTreeRegressor(max_depth=depth)
#     temperatureTree.fit(xTrain, yTrain[:, 0])
#     windTree.fit(xTrain, yTrain[:, 2])
#     precipTree.fit(xTrain, yTrain[:, 3])
#     if np.sqrt(np.mean((temperatureTree.predict(xTest)-yTest[:, 0])**2)) < tempMin:
#         bestTree[0] = temperatureTree
#         tempMin = np.sqrt(np.mean((temperatureTree.predict(xTest)-yTest[:, 0])**2))
#         print("Temp:", depth)
#     if np.sqrt(np.mean((windTree.predict(xTest)-yTest[:, 2])**2)) < windMin:
#         bestTree[1] = windTree
#         windMin = np.sqrt(np.mean((windTree.predict(xTest)-yTest[:, 2])**2))
#         print("Wind:", depth)
#     if np.sqrt(np.mean((precipTree.predict(xTest)-yTest[:, 3])**2)) < precipMin:
#         bestTree[2] = precipTree
#         precipMin = np.sqrt(np.mean((precipTree.predict(xTest)-yTest[:, 3])**2))
#         print("Precip:", depth)
#     rmse["temp"].append(np.sqrt(np.mean((temperatureTree.predict(xTrain)-yTrain[:, 0])**2)))
#     rmse["temp"].append(np.sqrt(np.mean((temperatureTree.predict(xTest)-yTest[:, 0])**2)))
#     rmse["wind"].append(np.sqrt(np.mean((windTree.predict(xTrain)-yTrain[:, 2])**2)))
#     rmse["wind"].append(np.sqrt(np.mean((windTree.predict(xTest)-yTest[:, 2])**2)))
#     rmse["precip"].append(np.sqrt(np.mean((precipTree.predict(xTrain)-yTrain[:, 3])**2)))
#     rmse["precip"].append(np.sqrt(np.mean((precipTree.predict(xTest)-yTest[:, 3])**2)))
#     # best trees
#     # temp: 11
#     # wind: 13
#     # precip: 7
# pickle.dump(bestTree[0], open("TemperatureTree", "wb"))
# pickle.dump(bestTree[1], open("WindTree", "wb"))
# pickle.dump(bestTree[2], open("PrecipTree", "wb"))

# fig, axis = plt.subplots(1, 3)
# axis[0].plot(range(start, end), rmse["temp"][::2])
# axis[0].plot(range(start, end), rmse["temp"][1::2])
# axis[0].set_title("Temperature")
# axis[0].set_xlabel("Depth of tree")
# axis[0].set_ylabel("RMSE")

# axis[1].plot(range(start, end), rmse["wind"][::2])
# axis[1].plot(range(start, end), rmse["wind"][1::2])
# axis[1].set_title("Wind speed")
# axis[1].set_xlabel("Depth of tree")

# axis[2].plot(range(start, end), rmse["precip"][::2])
# axis[2].plot(range(start, end), rmse["precip"][1::2])
# axis[2].set_title("Precipitation")
# axis[2].set_xlabel("Depth of tree")

# axis[0].grid()
# axis[1].grid()
# axis[2].grid()
# plt.savefig("trees.pdf", format="pdf", bbox_inches="tight")
# plt.show()
