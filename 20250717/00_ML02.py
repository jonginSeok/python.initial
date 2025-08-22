'''
2025.07.17

노이즈
'''

# 1. 라이브러리 임포트
from sklearn.datasets import make_classification
import joblib  # joblib은 sklearn 모델/객체 저장에 최적화
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import fetch_california_housing
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 2. 데이터 생성
np.random.seed(42)
x_np = np.linspace(-5, 5, 200)
y_np = 3 * x_np**2 + 2 + np.random.normal(0, 5, size=x_np.shape)

# 3. 훈련/검증 데이터 분리 (Overfitting 감지를 위해)
x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(
    x_np, y_np, test_size=0.2, random_state=42)

x_train = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
x_val = torch.tensor(x_val_np, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)

# 4. 모델 정의
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),          # 위의 레이어에 포함된 각 노드에 연결됨
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# 5. 손실 함수, 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. 학습
epochs = 10000
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()  # 자동 미분
    optimizer.step()  # 가중치 갱신
    train_losses.append(loss.item())

    # 검증 손실 계산
    model.eval()
    with torch.no_grad():  # 학습하지 않는다. optimizer.zero_grad()
        val_pred = model(x_val)
        val_loss = criterion(val_pred, y_val)
        val_losses.append(val_loss.item())

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# 7. Loss 시각화 (Overfitting 감지)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.show()

# 8. 모델 저장
torch.save(model.state_dict(), "quadratic_model.pth")
print(" 모델이 'quadratic_model.pth'로 저장되었습니다.")

# 9. 모델 새로 로드
loaded_model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
loaded_model.load_state_dict(torch.load("quadratic_model.pth"))
loaded_model.eval()
print(" 저장된 모델을 성공적으로 로드했습니다.")

# 10. 예측 시각화
x_test = torch.linspace(-5, 5, 100).unsqueeze(1)  # 2번째 차원 추가
with torch.no_grad():
    y_test_pred = loaded_model(x_test).squeeze().numpy()

plt.scatter(x_np, y_np, label='Original Data', alpha=0.6)
plt.plot(x_test.squeeze().numpy(), y_test_pred,
            color='red', label='Model Prediction')
plt.title("Model Fit to Quadratic Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# 11. 모델 사용 예제
x_input = torch.tensor([[4.0]])
with torch.no_grad():
    y_output = loaded_model(x_input)
print(f" Predicted y for x=4.0: {y_output.item():.4f}")


#!pip install torchsummary

summary(model, input_size=(1,))


# Multivariate Linear Regression
# y = 3 * x1​ + 2 * x2 + 1 + noise

# 1. 데이터 생성
torch.manual_seed(42)
n_samples = 100
x = torch.randn(n_samples, 2)  # 100 x 2 입력: x1, x2
true_w = torch.tensor([3.0, 2.0])  # 가중치
true_b = 1.0  # 편향

# 타겟 y 계산 (노이즈 추가), @:행렬곱(행과 열 벡터를 내적)
y = x @ true_w + true_b + 0.1 * torch.randn(n_samples)

# 2. 선형 회귀 모델 정의
model = nn.Linear(2, 1)  # 입력 특성 2개, 출력 1개

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
    loss.backward()
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


# 데이터셋 로드
data = fetch_california_housing(as_frame=True)
df = data.frame

# CSV 저장 (선택사항)
df.to_csv("california_housing.csv", index=False)
print(df.head())


# CSV 파일 로드
df = pd.read_csv("california_housing.csv")

# 특성과 타겟 선택 (다변수 가능)
X = df[['AveRooms', 'AveOccup', 'HouseAge']].values  # 3개의 특성
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


# 예측 수행
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()              # 정규화된 예측
    y_true = y_tensor.numpy()                     # 정규화된 실제값

# 역정규화 (원래 단위로)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_true_inv = scaler_y.inverse_transform(y_true)

# 평가 지표 계산
mse = mean_squared_error(y_true_inv, y_pred_inv)
mae = mean_absolute_error(y_true_inv, y_pred_inv)
r2 = r2_score(y_true_inv, y_pred_inv)   # 결정계수(0~1)

print("\n 모델 평가 지표:")
print(f"MSE (평균 제곱 오차):      {mse:.4f}")
print(f"MAE (평균 절대 오차):      {mae:.4f}")
print(f"R² Score (설명력):         {r2:.4f}")  # 모든 feature 사용 시 더 높아짐


# 예측
# 새로운 입력값 (예: 방 5개, 평균 거주인원 3명, 집 나이 20년)
new_data = [[5.0, 3.0, 20.0]]  # shape: [1, 3]

# 1. 입력 정규화
new_data_scaled = scaler_X.transform(new_data)
new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# 2. 모델 예측
model.eval()  # 평가 모드로 전환 (필수는 아니지만 관례적으로)
with torch.no_grad():
    prediction = model(new_tensor)  # shape: [1, 1]

# 3. 예측값 역정규화
pred_value = scaler_y.inverse_transform(prediction.numpy())[0][0]

print(f"예측된 집값: {pred_value:.3f} (단위: 10만 달러)")


# 모델과 스케일러 저장 (필요 시 함께 저장하는 것이 중요)
torch.save(model.state_dict(), "linear_model.pth")

# Scaler도 함께 저장 (입출력 정규화 복원용)
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print(" 모델과 스케일러 저장 완료")


# 모델 로드
# 모델 구조 재정의 (학습 당시와 동일해야 함)
loaded_model = nn.Linear(3, 1)  # 입력 특성 3개
loaded_model.load_state_dict(torch.load("linear_model.pth"))
loaded_model.eval()  # 평가 모드 전환

# 스케일러 불러오기
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

print(" 모델과 스케일러 로드 완료")


# 로드된 모델을 사용하여 예측하기
# 새 데이터: 방 4개, 평균 거주 2.5명, 집 나이 15년
new_data = [[4.0, 2.5, 15.0]]

# 입력 정규화
new_scaled = scaler_X.transform(new_data)
new_tensor = torch.tensor(new_scaled, dtype=torch.float32)

# 예측
with torch.no_grad():
    pred = loaded_model(new_tensor)

# 결과 역정규화
pred_value = scaler_y.inverse_transform(pred.numpy())[0][0]

print(f"예측된 집값: {pred_value:.3f} (단위: 10만 달러)")


# 1. 데이터 생성 (2진 분류용)
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
                            # 실제로 분류에 영향을 주는 의미 있는(feature informative) 특성 수
                            n_informative=2,
                            n_redundant=0,   # 쓸모 없는 특성(중복된 정보) 수
                            random_state=0)

# 2. train/val 분리 및 정규화
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

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
    nn.ReLU(),
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
        print(
            f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

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
