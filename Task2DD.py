#10fold cross validation with ridge regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = np.genfromtxt('train.csv', delimiter=",")

#slice out the X values as an X matrix
train_X = train[1:,1:]
train_y = train[1:,0]

print(np.size(train_y))
print(np.size(train_X))


regressor1 = RidgeCV(alphas =[0.1, 1, 10, 100, 200], cv=10)     #is this so precise?
regressor1.fit(train_X, train_y)


Y_pred =regressor1.predict(train_X)
RMSE = mean_squared_error(train_y, Y_pred)**0.5     # why hoch 2?

print(RMSE)
#print(regressor1.)
#print(regressor1.predict(train_X))          #predict using the linear model.
#print(regressor1.score(train_X, train_y))
#print(train_y)
