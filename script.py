import pandas as pd
import sklearn.linear_model as linear_model
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import

data = pd.read_csv("tourism_data_500_points.csv", sep=";")

X_train, X_test, y_train, y_test = 

model_lasso = linear_model.Lasso(alpha=0.5)
model_RF = RandomForestRegressor(max_depth=2, random_state=0)

model_lasso.fit(X_train, y_train)
model_RF.fit(X_train, y_train)
