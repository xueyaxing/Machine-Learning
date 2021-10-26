import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import sklearn.pipeline as pl
import sklearn.preprocessing as sp

data = pd.read_csv('Salary.csv')
x = pd.DataFrame(data['YearsExperence'])
y = data['Salary']
model = pl.make_pipeline(sp.PolynomialFeatures(20),lm.Ridge())
model.fit(x,y)
pred_test_y = model.predict(x)
print(sm.r2_score(y,pred_test_y))

plt.grid(linestyle=':')
plt.scatter(data['YearsExperence'], y, s=60, color='dodgerblue', label='Samples')
plt.plot(data['YearsExperence'], pred_test_y, color='orangered', label='Regression Line')
plt.legend()
plt.show()
