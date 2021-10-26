import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sd

boston = sd.load_boston()
# 把数据存入dataframe
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['TARGET'] = boston.target

# 对某些字段进行简单的数据分析
# data.plot.scatter(x='RM',y='TARGET')

# 整理输入集 输出集 拆分测试集 训练集
import  sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as sm
x, y = data.iloc[:, :-1], data['TARGET']
# random_state随机种子 对同一组数据使用相同的随机种子划分数据集，得到的结果是一样的
train_x, test_x, train_y, test_y =ms.train_test_split(x, y, test_size=0.2, random_state=5)
# 训练模型-线性回归
model = lm.LinearRegression()
model.fit(train_x, train_y)
# 针对测试样本测试，评估
pred_test_y = model.predict(test_x)
print('线性回归-',sm.r2_score(test_y,pred_test_y))

# 训练模型-岭回归
params = np.arange(1,500,1)
for param in params:
    model = lm.Ridge(param)
    model.fit(train_x, train_y)
    # 针对测试样本测试，评估
    pred_test_y = model.predict(test_x)
    print('岭回归-',sm.r2_score(test_y,pred_test_y))

# 训练模型-多项式回归
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
model = pl.make_pipeline(sp.PolynomialFeatures(2),lm.Ridge())
model.fit(train_x,train_y)
pred_test_y = model.predict(test_x)
print('多项式回归-',sm.r2_score(test_y,pred_test_y))
