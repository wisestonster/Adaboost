# 에다부스트로 악성/양성 유방암 분류하기
# https://www.codeit.kr/learn/courses/machine-learning/3133
# Adaboost 설명: 
# - root node와 분류노드 2개를 갖는 얕은 결정 트리(stump)만을 사용
# - 다른 학습 알고리즘(약한 학습기, weak learner)의 결과물들을 가중치를 두어 더하는 방법 -> 틀리게 예측한 데이터의 중요도를 올려주고, 맞게 예측한 데이터 중요도를 낮춰준다.
# - 최종 결정을 내릴 때, 성능이 좋은 결정 스텀프들 예측 의견의 비중은 높고, 그렇지 않은 결정 스텀프의 의견의 비중은 낮게 반영합니다.
# - 참고: https://ko.wikipedia.org/wiki/%EC%97%90%EC%9D%B4%EB%8B%A4%EB%B6%80%EC%8A%A4%ED%8A%B8

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd

# 데이터 셋 불러 오기
cancer_data = load_breast_cancer()
# 데이터 셋을 살펴보기 위한 코드
"""print(cancer_data.DESCR)"""

# 데이터프레임으로 만들기
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])

# 데이터셋 나누기 (n_estimator는 생성할 tree의 개수, 디폴트는 10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
model = AdaBoostClassifier(n_estimators=50, random_state=5)
model.fit(X_train, y_train)

# 테스트셋으로 검증
predictions = model.predict(X_test)
score = model.score(X_test, y_test)

# 출력 코드
print(predictions)
print(score)
