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

# 2. train/val 분리 및 정규화
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 3. 텐서 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (N, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# 4. 모델 정의 (출력 1개 + sigmoid는 생략)
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(), # 
    nn.Linear(16, 1)  # 마지막에 sigmoid는 BCEWithLogitsLoss가 내부에서 처리함
)

# 5. 손실 함수 + 옵티마이저
criterion = nn.BCEWithLogitsLoss()  # 손실함수 + Sigmoid
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. 학습
epochs = 100
train_loss_history = []
val_loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    train_loss_history.append(loss.item())

    # 검증
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        val_loss_history.append(val_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# 7. 시각화
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title("Binary Classification Loss (BCE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()

# 8. 정확도 계산
with torch.no_grad():
    probs = torch.sigmoid(model(X_val))  # 로짓 → 확률
    preds = (probs > 0.5).float()        # Tensor -> float
    acc = (preds == y_val).float().mean()
    print(f" Validation Accuracy: {acc.item()*100:.2f}%")
