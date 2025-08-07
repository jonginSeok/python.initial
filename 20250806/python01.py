import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

np.random.seed(42)

# X와 Y 생성
X = np.linspace(0, 10, 100)
Y = 2 * X + 1 + np.random.normal(0, 1, size=X.shape)

df = pd.DataFrame({"X": X, "Y": Y})

# 시각화
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x="X", y="Y", data=df)
plt.title(" X vs Y Scatter")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# PyTorch 텐서로 변환
X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1)
Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 입력 1, 출력 1

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []

for epoch in range(2000):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

plt.plot(losses)
plt.title(" Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


model.eval()
with torch.no_grad():
    Y_pred = model(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(X_test.numpy(), Y_test.numpy(), label="Real", color="blue")
plt.scatter(X_test.numpy(), Y_pred.numpy(), label="Prediction", color="red")
plt.title(" Real vs Prediction (Test Set)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

X_all = torch.tensor(X, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    Y_all_pred = model(X_all)

plt.figure(figsize=(8, 5))
plt.scatter(X, Y, label="Real", color="gray")
plt.plot(X, Y_all_pred.numpy(), color="red", label="Prediction", linewidth=2)
plt.title(" Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


# 모델 저장
torch.save(model.state_dict(), "linear_model.pth")
print("모델 저장 완료: linear_model.pth")


# 저장된 모델 불러오기
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load("linear_model.pth"))
loaded_model.eval()
print("모델 로드 성공")


# 예측 및 시각화
sample = [[10]]
X_sample = torch.Tensor(sample)
with torch.no_grad():
    predicted = loaded_model(X_sample)
    print(predicted)
