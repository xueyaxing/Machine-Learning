import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sd
import sklearn.metrics as sm
import sklearn.model_selection as ms
import sklearn.linear_model as lm
iris = sd.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

#基于透视表完成简单数据统计分析
# print(data.pivot_table(index='target'))

#可视化
# data.plot.scatter(x='petal length (cm)', y='petal width (cm)', c='target', cmap='brg')
# data.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', c='target', cmap='brg', s=20)
# plt.show()

#逻辑回归器
"""
    solver:用来指明损失函数的优化方法，sklearn自带了如下几种：
        liblinear:坐标轴下降法来迭代优化损失函数
        newton-cg:牛顿法的一种
        lbfgs: 拟牛顿法
        sag: 随机平均梯度下降 (适合样本量大的情况)
    penalty: 参数可以选l1 和 l2 ，与solver有关
        L2正则化，所有优化算法都适用
        l1正则化，只能使用liblinear
    C: 该参数可以控制正则强度，值越小，正则强度越大，可以防止过拟合
"""
#
#取消部分样本，选用全样本集
#整理输入和输出集，拆分测试集 训练集
x, y = data.iloc[:, :-1], data['target']
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.1, random_state=7, stratify=y)
model = lm.LogisticRegression()

#做五次交叉验证
scores = ms.cross_val_score(model, x, y, cv=5, scoring='accuracy')
print(scores.mean())

model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
#模型准确率
print((pred_test_y==test_y).sum()/test_y.size)
print(test_y.values)