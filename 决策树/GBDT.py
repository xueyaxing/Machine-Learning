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

#GBDT
import sklearn.ensemble as se
model = se.GradientBoostingRegressor(max_depth=5, n_estimators=50, random_state=7, min_samples_split=3)
model.fit(train_x, train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y, pred_test_y))
