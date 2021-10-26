import pandas as pd
import sklearn.tree as st
import sklearn.datasets as sd
import sklearn.metrics as sm
import sklearn.model_selection as ms
boston = sd.load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['TARGET'] = boston.target
x, y = data.iloc[:, :-1], data['TARGET']
train_x, test_x, train_y, test_y = ms.train_test_split(x, y, test_size=0.2, random_state=7)
model = st.DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)

#测试特征重要性
fi = model.feature_importances_
# print('fi',fi,train_x.columns)
s = pd.Series(fi, index=train_x.columns)
print(s)

pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))

