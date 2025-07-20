# Multivariate Linear Regression
# y = 3 * x1​ + 2 * x2 + 1 + noise
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 데이터 생성
torch.manual_seed(42) # 각자 변경한다!!!
n_samples = 100
x = torch.randn(n_samples, 2)  # 100 x 2 입력: x1, x2
true_w = torch.tensor([3.0, 2.0])  # 가중치
true_b = 1.0  # 편향

# 타겟 y 계산 (노이즈 추가), @:행렬곱(행과 열 벡터를 내적)
y = x @ true_w + true_b + 0.1 * torch.randn(n_samples)

# 2. 선형 회귀 모델 정의
model = nn.Linear(2, 1)  # 입력 특성 2개, 출력 1개 (선형회귀식 생성)

# 3. 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 4. 학습 루프
loss_history = []
epochs = 100
for epoch in range(epochs):
    y_pred = model(x).squeeze()  # (100, 1) → (100,)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward() # CPU, GPU 모두 지원, 사용 선택???
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}")

# 5. 학습된 파라미터 출력
learned_w = model.weight.data
learned_b = model.bias.data
print(f"\nLearned weights: {learned_w}")
print(f"Learned bias: {learned_b}")

# 6. 손실 시각화
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

# 7. 학습된 모델 사용 예제
# 새로운 입력 샘플: x1 = 1.0, x2 = 2.0
new_x = torch.tensor([[1.0, 2.0]])  # shape: (1, 2)

# 모델을 사용하여 예측
model.eval()  # (선택) 추론 모드로 설정
with torch.no_grad():
    predicted_y = model(new_x)

print(f"\n New input: x1=1.0, x2=2.0 → Predicted y: {predicted_y.item():.4f}")