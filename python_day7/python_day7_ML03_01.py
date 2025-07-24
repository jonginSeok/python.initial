#ML03.txt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 데이터 생성 (클래스 3개)
X, y = make_classification(n_samples=1000, n_features=4, n_classes=3,
                           n_informative=3, n_redundant=0, random_state=42)
# n_features : 분류 특성 수
# n_informative : 분류에 영향을 미치는 컬럼 수
# n_redundant : 의미 중복 컬럼 수

# 2. 학습/검증 분리
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42) # 20% / 80%

# 3. 정규화(데이터를 정제하여 '표준정규분포'(평균:0, 표준편차 :1)화하여 빠른 학습(출력)을 유도)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # 평균:0, 표준편차 :1
X_valid = scaler.transform(X_valid)

# Criterion: ① 기준 ② 평가 ③ 심사 ④ 조건
# Accuracy: ① 정확도 ② 정확 ③ 면밀

# under-fitting
# over-fitting

# 4. 텐서 변환(X_train, X_valid가 nparray이여서 변환)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)   # CrossEntropyLoss는 long 타입 정수
y_valid = torch.tensor(y_valid, dtype=torch.long)

# 5. 모델 정의 (출력 노드 수 = 클래스 수)
model = nn.Sequential(
    nn.Linear(4, 16),   # 16 : 가중치(낮게 설정하여 조금씩 올리며 학습(출력)이 좋은 수치 판단) , n_features=4
    nn.ReLU(),          # 비선형 분류
    nn.Linear(16, 16),  # add ngins7512
    nn.ReLU(),
    nn.Linear(16, 16),  # add ngins7512
    nn.ReLU(),
    nn.Linear(16, 3)  # 클래스 수 = 3 , n_classes=3
)

# 6. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()  # 내부에 softmax 포함 # 다중
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam : 최근 알고리즘, 경사하강법, 대표적 사용

# 7. 학습
epochs = 100
train_losses = []
valid_losses = []

for epoch in range(epochs):
    model.train()         # 생략가능
    optimizer.zero_grad() # 미분 초기화
    output = model(X_train)            # shape: (batch_size, 3)
    loss = criterion(output, y_train)  # CrossEntropyLoss expects raw logits + long labels
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 검증
    model.eval() # eval하기 때문에 model.train()를 적어준다. # 계산과 관련없음
    with torch.no_grad(): # 학습하지 않는다. optimizer.zero_grad()
        valid_output = model(X_valid)
        valid_loss = criterion(valid_output, y_valid)
        valid_losses.append(valid_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Valid Loss = {valid_loss.item():.4f}")

# 8. 시각화
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Valid Loss") # 과적합(?)하면 valid_losses 수치 올라감.
plt.title("Multiclass Classification Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# 9. 정확도 평가
with torch.no_grad():
    logits = model(X_valid)     # Softmax를 적용하지 않았으므로 확률이 아닌 값 리턴
    preds = torch.argmax(logits, dim=1)    # 가장 높은 확률의 클래스 인덱스 리턴   # dim=1: 1차원, 1행
    acc = (preds == y_valid).float().mean()  # 정답을 대상으로 평균 계산
    print(f" Validation Accuracy: {acc.item() * 100:.2f}%")
