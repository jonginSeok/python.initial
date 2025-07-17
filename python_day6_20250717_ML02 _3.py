import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # 선형회귀에서는 필수!

# CSV 파일 로드
df = pd.read_csv("california_housing.csv")

# 특성과 타겟 선택 (다변수 가능)
X = df[['AveRooms', 'AveOccup', 'HouseAge']].values  # 3개의 특성, X(대문자:다차원)
y = df['MedHouseVal'].values  # 타겟: 집값

# 정규화 (중요!), 평균=0, 표준편차=1, 빠른 수렴과 안정적인 학습
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# PyTorch tensor 변환
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 모델 정의
model = nn.Linear(X_tensor.shape[1], 1)  # 디폴트 모드 = train

# 손실 함수 및 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습
epochs = 2000
losses = []
for epoch in range(epochs):
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 학습된 파라미터 출력
print("\nWeights:", model.weight.data)
print("Bias:", model.bias.data)

# 손실 시각화
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()