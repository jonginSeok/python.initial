"""
2025.07.16
"""

# y = w * x + b
# 직선의 방정식과 기울기(계수, 가중치, weight), 절편(편견,편향, bias)
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

# 파라미터 설정
w = 0.0000001
b = 3

# x 값 100개 생성 (예: -10부터 10까지 균등 분포)
x = np.linspace(-10, 10, 100)

# y 값 계산
y = w * x + b

print("x=", x)
print("y=", y)

# 시각화
plt.figure(figsize=(5, 3))
plt.plot(x, y, label=f"y = {w}x + {b}", color="blue")
plt.title(f"Linear Function: y = {w}x + {b}")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")  # x축과 y축의 비율을 동일하게 설정
plt.grid(True)
plt.legend()
plt.show()


# y = wx + b
# 15 = w * 2 + b
# 초기값
w = 0.0
b = 0.0

# 학습 데이터
x = 2
y = 15

# 학습률
lr = 0.01

# 에포크 수
epochs = 200

for epoch in range(epochs):
    # 순전파 (Forward)
    y_pred = w * x + b
    loss = 0.5 * (y_pred - y) ** 2  # MSE 손실

    # 역전파 (Backward / Gradient:기울기)
    dL_dy_pred = y_pred - y  # dL/dy_pred
    # Chain rule: dL/dw = (dy_pred/dw) * (dL/dy_pred)
    dL_dw = dL_dy_pred * x
    # Chain rule: dL/db = (dy_pred/db) * (dL/dy_pred)
    dL_db = dL_dy_pred * 1

    # 파라미터 업데이트 (Gradient Descent)
    w -= lr * dL_dw
    b -= lr * dL_db

    # 10회마다 출력
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")


# y = 2x^2 + 3x + 1
# 2차 방정식과 가중치, 편향값 시각화

# 파라미터 설정
w1 = 2
w2 = 3
b = 1

# x 값 100개 생성 (예: -5부터 5까지 균등 분포)
x = np.linspace(-5, 5, 100)

# y 값 계산
y = w1 * x**2 + w2 * x + b

print("x=", x)
print("y=", y)

# 시각화
plt.figure(figsize=(5, 3))
plt.plot(x, y, label=f"y = {w1} * x^2 + {w2} * x + {b}", color="blue")
plt.title(f"y = {w1} * x^2 + {w2} * x + {b}")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")  # x축과 y축의 비율을 동일하게 설정
plt.grid(True)
plt.legend()
plt.show()


# 아래의 식을 사용하여 가중치(w1, w2)와 편향값(b)을 결정할 때 편미분을 활용해보세요.
# x = 2
# 2*x**2 + 3*x + 1  # 120

x = 2
y = 15
w1 = 1
w2 = 1
b = 1

loss = 0.5 * (y_pred - y) ** 2  # MSE 손실

y_pred = w1 * x**2 + w2 * x + b
y_pred  # 7 -> 15


# 학습률
lr = 0.01

# 에포크 수
epochs = 50

for epoch in range(epochs):
    # 순전파 (Forward)
    y_pred = w1 * x**2 + w2 * x + b
    loss = 0.5 * (y_pred - y) ** 2  # MSE 손실

    # 역전파 (Backward / Gradient)
    dL_dy_pred = y_pred - y  # dL/dy_pred
    dL_dw1 = dL_dy_pred * x**2
    dL_dw2 = dL_dy_pred * x
    dL_db = dL_dy_pred * 1

    # 파라미터 업데이트 (Gradient Descent)
    w1 -= lr * dL_dw1
    w2 -= lr * dL_dw2
    b -= lr * dL_db

    # 10회마다 출력
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch:3d}: Loss = {loss:.4f}, w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}"
        )


# 학습 후
y_pred = w1 * x**2 + w2 * x + b
y_pred


# 1. Training data
x_train = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
y_train = torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)

# 2. Model definition (Linear Regression)
model = nn.Linear(in_features=1, out_features=1)

# 3. Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# For storing loss over epochs
loss_history = []

# 4. Training loop
epochs = 20
for epoch in range(epochs):
    # Forward pass
    y_pred = model(x_train)

    # Compute loss
    loss = criterion(y_pred, y_train)
    loss_history.append(loss.item())

    # Backward and optimization
    optimizer.zero_grad()
    loss.backward()  # 자동 미분
    optimizer.step()  # 가중치 갱신

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 5. Test prediction
test_input = torch.tensor([[6.0]])
predicted = model(test_input).item()  # 하나의 값만 들어 있을 때 가져오기
print(f"\nPredicted score for 6 hours of study: {predicted:.2f}")

# 6. Visualization
predicted_y = model(x_train).detach().numpy()  # Tensor를 계산 그래프에서 분리

plt.figure(figsize=(12, 5))

# 6-1. Plot data and regression line
plt.subplot(1, 2, 1)
plt.scatter(x_train.numpy(), y_train.numpy(), label="Actual Data")
plt.plot(x_train.numpy(), predicted_y, color="red", label="Model Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Study Time vs Exam Score")
plt.legend()
plt.grid(True)

# 6-2. Plot loss curve
plt.subplot(1, 2, 2)
plt.plot(loss_history, color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Training Epochs")
plt.grid(True)

plt.tight_layout()
plt.show()


x = np.linspace(-10, 10, 100)
y = 3 * x + 4

x_train = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
y_train = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

model = nn.Linear(in_features=1, out_features=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_history = []
epochs = 350
for epoch in range(epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss_history.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

pred = model(torch.tensor([[10.0]])).detach().numpy()
print(f"10 -> {pred[0][0]}")


plt.figure(figsize=(5, 3))
plt.plot(loss_history, color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Training Epochs")
plt.grid(True)
plt.show()


# 정규분포 노이즈 추가 (평균=0, 표준편차=2)
noise = np.random.normal(loc=0.0, scale=2.0, size=x.shape)
y_noisy = y + noise
