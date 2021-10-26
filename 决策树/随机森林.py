import pandas as pd
import sklearn.tree as st
import sklearn.datasets as sd
import sklearn.metrics as sm
import sklearn.model_selection as ms
boston = sd.load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['TARGET'] = boston.target
x, y = data.iloc[:, :-1], data['TARGET']
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.1, random_state=7)
model = st.DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))

#随机森林
#max_depth: 决策树最大深度
#n_estimators: 构建400课决策树，训练模型
#min_samples_split: 子表中最小样本数，若小于这个数字，将不会继续向下拆分
import sklearn.ensemble as se
model = se.RandomForestRegressor(max_depth=5, n_estimators=400, min_samples_split=3)
model.fit(train_x,train_y)
test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))