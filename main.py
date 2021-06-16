# https://www.codeit.kr/learn/courses/machine-learning/3133
# 12. 에다 부스트로 악성/양성 유방암 분류하기
# Adaboost 설명: https://ko.wikipedia.org/wiki/%EC%97%90%EC%9D%B4%EB%8B%A4%EB%B6%80%EC%8A%A4%ED%8A%B8
# 다른 학습 알고리즘(약한 학습기, weak learner)의 결과물들을 가중치를 두어 더하는 방법

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd

# 데이터 셋 불러 오기
cancer_data = load_breast_cancer()
# 데이터 셋을 살펴보기 위한 코드
"""print(cancer_data.DESCR)"""

X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
model = AdaBoostClassifier(n_estimators=50, random_state=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = model.score(X_test, y_test)
# 코드를 쓰세요

# 출력 코드
print(predictions)
print(score)