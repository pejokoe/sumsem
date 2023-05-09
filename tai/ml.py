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

# trees(xTrain, yTrain)
# randomForest(xTrain, yTrain)
# neuralNet(xTrain, yTrain)

# create best models
models = "tempTree windTree precipTree tempForest windForest precipForest tempNN windNN precipNN".split()
rmses = []
# trees
# tempTree = tree.DecisionTreeRegressor(max_depth=11)
# tempTree.fit(xTrain, yTrain[:, 0])
# rmses.append(mean_squared_error(tempTree.predict(xTest), yTest[:, 0])**0.5)
# windTree = tree.DecisionTreeRegressor(max_depth=13)
# windTree.fit(xTrain, yTrain[:, 2])
# rmses.append(mean_squared_error(windTree.predict(xTest), yTest[:, 2])**0.5)
# precipTree = tree.DecisionTreeRegressor(max_depth=7)
# precipTree.fit(xTrain, yTrain[:, 3])
# rmses.append(mean_squared_error(precipTree.predict(xTest), yTest[:, 3])**0.5)
# print("done with trees")

# # random forests
# tempForest = RandomForestRegressor(75, max_depth=11)
# tempForest.fit(xTrain, yTrain[:, 0])
# rmses.append(mean_squared_error(tempForest.predict(xTest), yTest[:, 0])**0.5)
# windForest = RandomForestRegressor(85, max_depth=13)
# windForest.fit(xTrain, yTrain[:, 2])
# rmses.append(mean_squared_error(windForest.predict(xTest), yTest[:, 2])**0.5)
# precipForest = RandomForestRegressor(35, max_depth=7)
# precipForest.fit(xTrain, yTrain[:, 3])
# rmses.append(mean_squared_error(precipForest.predict(xTest), yTest[:, 3])**0.5)
# print("done with forests")

# # neural net
# nn = MLPRegressor([9, 6], max_iter=18)
# nn.fit(xTrain, yTrain[:, [0, 2, 3]])
# rmses.append(mean_squared_error(nn.predict(xTest)[:, 0], yTest[:, 0])**0.5)
# rmses.append(mean_squared_error(nn.predict(xTest)[:, 1], yTest[:, 2])**0.5)
# rmses.append(mean_squared_error(nn.predict(xTest)[:, 2], yTest[:, 3])**0.5)

# # saving models
# pickle.dump(tempTree, open("tempTree", "wb"))
# pickle.dump(windTree, open("windTree", "wb"))
# pickle.dump(precipTree, open("precipTree", "wb"))
# pickle.dump(tempForest, open("tempForest", "wb"))
# pickle.dump(windForest, open("windForest", "wb"))
# pickle.dump(precipForest, open("precipForest", "wb"))
# pickle.dump(nn, open("nn", "wb"))

# load models
tempTree = pickle.load(open("tempTree", "rb"))
windTree = pickle.load(open("windTree", "rb"))
precipTree = pickle.load(open("precipTree", "rb"))
tempForest = pickle.load(open("tempForest", "rb"))
windForest = pickle.load(open("windForest", "rb"))
precipForest = pickle.load(open("precipForest", "rb"))
nn = pickle.load(open("nn", "rb"))
rmses.append(mean_squared_error(tempTree.predict(xTest), yTest[:, 0])**0.5)
rmses.append(mean_squared_error(windTree.predict(xTest), yTest[:, 2])**0.5)
rmses.append(mean_squared_error(precipTree.predict(xTest), yTest[:, 3])**0.5)
rmses.append(mean_squared_error(tempForest.predict(xTest), yTest[:, 0])**0.5)
rmses.append(mean_squared_error(windForest.predict(xTest), yTest[:, 2])**0.5)
rmses.append(mean_squared_error(precipForest.predict(xTest), yTest[:, 3])**0.5)
rmses.append(mean_squared_error(nn.predict(xTest)[:, 0], yTest[:, 0])**0.5)
rmses.append(mean_squared_error(nn.predict(xTest)[:, 1], yTest[:, 2])**0.5)
rmses.append(mean_squared_error(nn.predict(xTest)[:, 2], yTest[:, 3])**0.5)
#plot rmses 
fig = plt.figure()
# ax = fig.add_axes()
plt.bar(models, rmses)
plt.title("RMSEs model comparison")
# ax.set_xlabels(models)
plt.grid()
plt.show()


