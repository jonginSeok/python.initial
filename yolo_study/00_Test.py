import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
    loss.backward()   # 자동 미분
    optimizer.step()  # 가중치 갱신

    # Print progress
    if epoch % 10 == 0:
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
plt.plot(x_train.numpy(), predicted_y, color='red', label="Model Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Study Time vs Exam Score")
plt.legend()
plt.grid(True)

# 6-2. Plot loss curve
plt.subplot(1, 2, 2)
plt.plot(loss_history, color='green')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Training Epochs")
plt.grid(True)

plt.tight_layout()
plt.show()


'''
x = np.linspace(-10, 10, 100)
y = 3*x + 4

x_train = torch.tensor(x.reshape(-1,1), dtype=torch.float32)
y_train = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

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
plt.plot(loss_history, color='green')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Training Epochs")
plt.grid(True)
plt.show()



# 정규분포 노이즈 추가 (평균=0, 표준편차=2)
noise = np.random.normal(loc=0.0, scale=2.0, size=x.shape)
y_noisy = y + noise

'''
