import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import pandas as pd

data = pd.read_csv('Salary.csv')
x = pd.DataFrame(data['YearsExperence'])
y = data['Salary']
model = lm.LinearRegression()
model.fit(x,y)
#针对训练样本执行预测操作，绘制回归线
pred_train_y = model.predict(x)

#可视化
plt.grid(linestyle=':')
plt.scatter(data['YearsExperence'],y,s=60,color='dodgerblue',label='Samples')
plt.plot(data['YearsExperence'],pred_train_y,color='orangered',label='Regression Line')
plt.legend()
plt.show()

#找到一组测试样本数据，输出评估指标结果
import sklearn.metrics as sm
test_x = x.iloc[::4]
test_y = y[::4]
pred_test_y = model.predict(test_x)
print( sm.mean_absolute_error(test_y,pred_test_y) )
print( sm.mean_squared_error(test_y,pred_test_y) )
print( sm.median_absolute_error(test_y,pred_test_y) )
print( sm.r2_score(test_y,pred_test_y))

#模型的保存
import pickle
with open('linear_model.pkl','wb') as f:
    pickle.dump(model,f)

#模型的加载
with open('linear_model.pkl','rb') as f:
    model = pickle.load(f)
print(model.predict([[15.6],[16.4]]))
