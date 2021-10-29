import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('cars.csv', header=None)

# 选择分类模型： 逻辑回归、决策树、RF、GBDT、AdaBoost
# 本次选择 RF(随机森林：相似的输入产生相似的输出)，也可以选择其他模型

# 数据预处理 针对当前数据完成标签编码 处理
import sklearn.preprocessing as sp
# 另存一份数据
train_data = pd.DataFrame([])
encoders = {}
for col_ind, col_val in data.items():
    encoder = sp.LabelEncoder()
    train_data[col_ind] = encoder.fit_transform(col_val)
    encoders[col_ind] = encoder
# 整理输入 与 输出集
x, y = train_data.iloc[:, :-1], train_data[6]
# 创建分类模型
import sklearn.ensemble as se
import sklearn.metrics as sm
import sklearn.model_selection as ms
model = se.RandomForestClassifier(max_depth=13, n_estimators=15, random_state=12)
# 学习曲线
params = np.arange(0.1,1.1,0.1)
_, train_scores, test_scores = ms.learning_curve(model, x, y, train_sizes=params, cv=5)
scores = test_scores.mean(axis=1)
plt.grid(linestyle=':')
plt.plot(params, scores, 'o-', color='green', label='Learning Curve', linestyle='--')
plt.legend()
plt.show()
# 做五次交叉验证
scores = ms.cross_val_score(model, x, y, cv=5, scoring='f1_weighted')
print(scores.mean())


# 验证曲线,选取最优超参数 n_estimators
# params = np.arange(15,16,1)
# train_scores, test_scores = ms.validation_curve(model, x, y, 'n_estimators', params, cv=5)
# scores = test_scores.mean(axis=1)
# 评估分数
# print(scores)
# 可视化
# import matplotlib.pyplot as plt
# plt.grid(linestyle=':')
# plt.plot(params, scores, 'o-', color='orangered', label='n_estimators VC', linestyle='--')
# plt.legend()
# plt.show()

# 验证曲线,选取最优超参数 n_estimators
# params = np.arange(13,14,1)
# train_scores, test_scores = ms.validation_curve(model, x, y, 'max_depth', params, cv=5)
# scores = test_scores.mean(axis=1)
# 评估分数
# print(scores)
# 可视化
# import matplotlib.pyplot as plt
# plt.grid(linestyle=':')
# plt.plot(params, scores, 'o-', color='dodgerblue', label='max_depth VC', linestyle='--')
# plt.legend()

# 验证曲线,选取最优超参数 random_state
# params = np.arange(1,20,1)
# train_scores, test_scores = ms.validation_curve(model, x, y, 'random_state', params, cv=5)
# scores = test_scores.mean(axis=1)
# 评估分数
# print(scores)
# 可视化
# import matplotlib.pyplot as plt
# plt.grid(linestyle=':')
# plt.plot(params, scores, 'o-', color='yellow', label='random_state VC', linestyle='--')
# plt.legend()
# plt.show()


model.fit(x, y)
pred_y = model.predict(x)
print(sm.classification_report(y, pred_y))

# 模型预测
data = [
    ['high', 'med', '5more', '4', 'big', 'low', 'unacc'],
    ['high', 'high', '4', '4', 'med', 'med', 'acc'],
    ['low', 'low', '2', '4', 'small', 'high', 'good'],
    ['low', 'med', '3', '4', 'med', 'high', 'vgood']
]
test_data = pd.DataFrame(data)
for col_ind, col_val in test_data.items():
    encoder = encoders[col_ind]
    encoded_col = encoder.transform(col_val)
    test_data[col_ind] = encoded_col

test_x, test_y = test_data.iloc[:, :-1], test_data[6]
pred_y = model.predict(test_x)
print(encoders[6].inverse_transform(pred_y))