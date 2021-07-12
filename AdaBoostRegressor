import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostRegressor

##########데이터 로드

x_data = np.array([
    [2, 1],
    [3, 2],
    [3, 4],
    [5, 5],
    [7, 5],
    [2, 5],
    [8, 9],
    [9, 10],
    [6, 12],
    [9, 2],
    [6, 10],
    [2, 4]
])
y_data = np.array([3, 5, 7, 10, 12, 7, 13, 13, 12, 13, 12, 6])

##########데이터 분석

##########데이터 전처리

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777)

##########모델 생성

model = AdaBoostRegressor(base_estimator=Lasso())

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) 

print(model.score(x_test, y_test)) 

##########모델 예측

x_test = np.array([
    [4, 6]
])

y_predict = model.predict(x_test)

print(y_predict[0])
