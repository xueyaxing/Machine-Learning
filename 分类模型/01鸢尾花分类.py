import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as sd
iris = sd.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

#基于透视表完成简单数据统计分析
# print(data.pivot_table(index='target'))

#可视化
data.plot.scatter(x='petal length (cm)', y='petal width (cm)', c='target', cmap='brg')
data.plot.scatter(x='sepal length (cm)', y='sepal width (cm)', c='target', cmap='brg', s=20)
plt.show()

