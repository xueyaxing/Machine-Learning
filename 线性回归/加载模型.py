import numpy as np
import pandas as pd
import pickle

#声明一个薪资预测类型，封装预测逻辑
class SalaryPredictionModel():
    def __init__(self):
        with open('linear_model.pkl','rb') as f:
            self.model = pickle.load(f)

    def predict(self,exps):
        """
        :param exps: 工作年限数组(一维数组)
        :return:
        """
        exps = np.array(exps).reshape(-1,1)
        return self.model.predict(exps)

model = SalaryPredictionModel()
print(model.predict([1,3.4,5,6,7,8,10]))