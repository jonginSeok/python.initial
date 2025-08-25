import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

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
