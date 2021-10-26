#共享单车投放量预测
import numpy as np
import pandas as pd
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.model_selection as ms
import sklearn.metrics as sm
import matplotlib.pyplot as plt
#加载数据集
data = pd.read_csv('Bike-Sharing_Day.csv')
data = data.drop(['instant', 'dteday', 'casual', 'registered'], axis=1)

#整理数据集，输入与输出集，训练集与测试集
x, y = data.iloc[:, :-1], data['cnt']
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.1, random_state=7)

#训练模型
model = se.RandomForestRegressor(max_depth=10, n_estimators=1000, min_samples_split=5)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
