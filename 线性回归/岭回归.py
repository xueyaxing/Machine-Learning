import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sklearn.linear_model as lm
import sklearn.metrics as sm
data = pd.read_csv('Salary2.csv')
x = pd.DataFrame(data['YearsExperence'])
y = data['Salary']

model = lm.LinearRegression()
model.fit(x, y)
pred_train_y = model.predict(x)
plt.grid(linestyle=':')
plt.scatter(data['YearsExperence'], y, s=60, color='dodgerblue', label='Samples')
plt.plot(data['YearsExperence'], pred_train_y, color='orangered', label='Regression Line')
plt.legend()
# plt.show()

#训练岭回归模型
model = lm.Ridge(500)
model.fit(x,y)
pred_train_y = model.predict(x)
plt.grid(linestyle=':')
plt.scatter(data['YearsExperence'], y, s=60, color='dodgerblue', label='Samples')
plt.plot(data['YearsExperence'], pred_train_y, color='orangered', label='Regression Line')
plt.legend()
plt.show()

#调整岭回归的参数
params = np.arange(60,130,1)
for param in params:
    model = lm.Ridge(param)
    model.fit(x,y)
    test_x , test_y = x.iloc[:30:4] , y[:30:4]
    pred_test_y = model.predict(test_x)
    print(param , "-->" ,sm.r2_score(test_y,pred_test_y))

