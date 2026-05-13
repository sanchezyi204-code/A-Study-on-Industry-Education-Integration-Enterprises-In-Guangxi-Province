# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 22:05:41 2025

@author: Friday
"""

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from category_encoders.target_encoder import TargetEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

df = pd.read_excel(r"")



x = df[['year','capital','employee_num','type','industry','partner','partner_num']]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

categorical_cols = ['type', 'industry','partner']
enc = TargetEncoder(cols=categorical_cols)
X_train[categorical_cols] = enc.fit_transform(X_train[categorical_cols], y_train)
X_test[categorical_cols] = enc.transform(X_test[categorical_cols])

rf = RandomForestRegressor(n_estimators=400, 
                           random_state=423,
                           max_depth=8,
                           min_samples_leaf=2,
                          )
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


