import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

arr = pd.read_csv("train.csv").values
X, y = arr[:, 1:], arr[:, 0]

def ridge_lambda(a):
    ridge = linear_model.Ridge(alpha = a)
    scores = cross_val_score(ridge, X, y, cv = 10, scoring = "neg_root_mean_squared_error")
    return np.average(np.negative(scores))

arr_lambda = [0.1, 1.0, 10.0, 100.0, 200.0]
df = pd.DataFrame({"ARMSE" :[ridge_lambda(i) for i in arr_lambda]})
df.to_csv("armse.csv", header = False, index = False)
    
    
    
