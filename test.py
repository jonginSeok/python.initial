import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 데이터 생성 (2진 분류용)
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
                           n_informative=2, # 실제로 분류에 영향을 주는 의미 있는(feature informative) 특성 수
                           n_redundant=0,   # 쓸모 없는 특성(중복된 정보) 수
                           random_state=0)

print(f' type:{type(X[:5])} \n{X[:5]}' )
print(f' type:{type(y[:5])} \n{y[:5]}' )
