# y = 3x^3 + 2x^2 + x +5 

# 1. 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split # 학습용/검증용 분할

from torchsummary import summary
# from torchinfo import summary  # torchsummary 대신 사용할 수 있음

# pip install torch
# pip install pillow
# pip install matplotlab


# pip install torchsummary
# pip install matplotlib
# pip install numpy
# pip install torch
# pip install scikit-learn

# 2. 데이터 생성
np.random.seed(0) # 무작위 seed 값 설정. seed 값이 같으면 결과가 같다.  42 -> 0

# x_np = np.linspace(-5, 5, 200) # x 는 문제 , y 는 정답
x_np = np.linspace(-5, 5, 100) # x 는 문제 , y 는 정답
# y_np = 3 * x_np**2 + 2 + np.random.normal(0, 5, size=x_np.shape) # 노이즈가 심함.
y_np = 3 * x_np**3 + 2 * x_np**2 + 5 + np.random.normal(0, 0, size=x_np.shape) # 노이즈가 심함.

# 3. 훈련/검증 데이터 분리 (Overfitting 감지를 위해)
x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(x_np, y_np, test_size=0.2, random_state=0) #x_train_np.shape, x_val_np.shape, y_train_np.shape, y_val_np.shape
x_train = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1) # GPU, 역전파
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1) # nparray 는 64bit여서 형변환
x_val = torch.tensor(x_val_np, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)

# 4. 모델 정의
model = nn.Sequential(
    nn.Linear(1, 64),  # 입력 1개 , 식 : y_np = 3 * x_np**2 + 2, -> 1의 의미
    nn.ReLU(),         # 위의 레이어에 포함된 각 노드에 연결됨
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64), # https://www.bing.com/images/search?view=detailV2&ccid=2DWFdeh9&id=D4CD2A8598E69FA5959D1A4DDB0511075B688E19&thid=OIP.2DWFdeh9FLfdxYYr5bFtVgHaDu&mediaurl=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1200%2F1*ZafDv3VUm60Eh10OeJu1vw.png&exph=603&expw=1200&q=Activation+Functions+ReLU&simid=608018141626776315&FORM=IRPRST&ck=76D4DDE3DAEBDE707AE8ED2E373F6048&selectedIndex=1&itb=0&cw=1728&ch=861&ajaxhist=0&ajaxserp=0
    nn.ReLU(),         # Activation Function, ReLU : 양수만 추출, 비선형 효과
    nn.Linear(64, 64),
    nn.ReLU(),         
    nn.Linear(64, 64), #  복잡(한)하게 계산
    nn.ReLU(),
    nn.Linear(64, 1)   # 출력 1개, 1 -> 1의 의미`
)

# 5. 손실 함수, 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # SGD이후 버전

# 6. 학습
# epochs = 10000
epochs = 1000
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 검증 손실 계산
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = criterion(val_pred, y_val)
        val_losses.append(val_loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# 7. Loss 시각화 (Overfitting 감지)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.show()

filePath = "quadratic_model_02.pth"

# 8. 모델 저장
torch.save(model.state_dict(), filePath)
print(f" 모델이 '{filePath}'로 저장되었습니다.")

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
loaded_model.load_state_dict(torch.load(filePath))
loaded_model.eval()
print(" 저장된 모델을 성공적으로 로드했습니다.")

# 10. 예측 시각화
x_test = torch.linspace(-5, 5, 100).unsqueeze(1) # 2번째 차원 추가
with torch.no_grad():
    y_test_pred = loaded_model(x_test).squeeze().numpy()

plt.scatter(x_np, y_np, label='Original Data', alpha=0.6)
plt.plot(x_test.squeeze().numpy(), y_test_pred, color='red', label='Model Prediction')
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

summary(loaded_model, input_size=(1,))  # 모델 구조 요약 출력
print(" 모델 구조 요약을 출력했습니다.")
