'''
2025.07.18
'''

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

# 2. 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 4. 텐서 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)   # CrossEntropyLoss는 long 타입 정수
y_val = torch.tensor(y_val, dtype=torch.long)

# 5. 모델 정의 (출력 노드 수 = 클래스 수)
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)  # 클래스 수 = 3
)

# 6. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()  # 내부에 softmax 포함
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. 학습
epochs = 100
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)            # shape: (batch_size, 3)
    loss = criterion(output, y_train)  # CrossEntropyLoss expects raw logits + long labels
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 검증
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# 8. 시각화
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Multiclass Classification Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# 9. 정확도 평가
with torch.no_grad():
    logits = model(X_val)     # Softmax를 적용하지 않았으므로 확률이 아닌 값 리턴
    preds = torch.argmax(logits, dim=1)    # 가장 높은 확률의 클래스 인덱스 리턴
    acc = (preds == y_val).float().mean()  # 정답을 대상으로 평균 계산
    print(f" Validation Accuracy: {acc.item() * 100:.2f}%")




import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
iris = load_iris()
X = iris.data  # shape: [150, 4]
y = iris.target  # shape: [150,]

# 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 텐서 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 2. 모델 정의
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 출력 노드: 3개 (클래스 수)
        )

    def forward(self, x):
        return self.net(x)

model = IrisNet()

# 3. 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 학습
epochs = 300
loss_history = []
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}")

# 5. 평가
model.eval()
with torch.no_grad():
    pred_test = model(X_test_tensor)
    predicted = torch.argmax(pred_test, dim=1)
    accuracy = accuracy_score(y_test, predicted.numpy())

print(f"\n 테스트 정확도: {accuracy * 100:.2f}%")

# 6. 손실 시각화
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()



Wine 등급 분류
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(url, header=None)
df


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 데이터 로드 및 전처리
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(url, header=None)

X = df.drop(0, axis=1).values
y = df[0].values - 1  # 클래스 레이블을 0부터 시작하도록 조정

# 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)  # 범주형 데이터는 정규화하면 안됨

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3, stratify=y   # 층화추출
)

# 텐서 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 2. 모델 정의
class WineNet(nn.Module):
    def __init__(self):
        super(WineNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 출력 노드: 3개 (클래스 수)
        )

    def forward(self, x):
        return self.net(x)

model = WineNet()

# 3. 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 학습
epochs = 50
loss_history = []
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}")

# 5. 평가
model.eval()
with torch.no_grad():
    pred_test = model(X_test_tensor)
    predicted = torch.argmax(pred_test, dim=1)
    accuracy = accuracy_score(y_test, predicted.numpy())

print(f"\n 테스트 정확도: {accuracy * 100:.2f}%")

# 6. 손실 시각화
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# 7. 평가
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(y_test, preds.numpy())
    print(f"\n 테스트 정확도: {acc * 100:.2f}%")

# 8. 새로운 샘플 예측
new_sample = X_test[0].reshape(1, -1)  # 예시로 첫 번째 샘플
scaled_new_sample = scaler.transform(new_sample)
new_tensor = torch.tensor(scaled_new_sample, dtype=torch.float32)
with torch.no_grad():
    out = model(new_tensor)
    pred_class = torch.argmax(out, dim=1).item()
    print(f"\n예측 결과: 클래스 {pred_class} (정답: {y_test[0]})")
    
    
    
    
    
#Simple CNN
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 1. 하이퍼파라미터 및 설정
BATCH_SIZE = 4
EPOCHS = 100
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE=", DEVICE)

# 2. 데이터 전처리
# 이미지를 전처리(Preprocessing) 하기 위한 연속된 변환 작업(transform pipeline) 을 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),      # 이미지를 고정 크기로 설정
    transforms.ToTensor(),              # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize([0.5], [0.5])  # 빠르고 안정적인 학습을 위한 정규화(0~1 -> -1~1), (x-0.5)/0.5
])
data_path = "학습용이미지 경로"
train_dataset = datasets.ImageFolder(root=data_path+'/train', transform=transform)
valid_dataset = datasets.ImageFolder(root=data_path+'/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 모델이 순서에 영향을 받지 않도록 매 epoch마다 무작위로 섞는다
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False) # 데이터 순서 고정

class_names = train_dataset.classes  # ['cat', 'dog']

# 3. 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(3채널(RGB), 필터수, 필터크기, stride=1, padding=0)
            nn.Conv2d(3, 16, 3, padding=1),  # 128x128x3 -> 128x128x16, padding=1은 1픽셀 추가하여 출력크기 유지
            nn.ReLU(),
            nn.MaxPool2d(2),                # -> 64x64x16, 이미지 크기를 1/2로 축소(국소적 특징 요약)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # -> 32x32x32
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),  # 입력은 CNN에서 전달된 크기, 출력은 보통 64, 128, 256, 512 등
            nn.ReLU(),
            nn.Linear(128, 2)   # 최종 출력이 1이면 Sigmoid연결, 2이면 Softmax연결
            # BCEWithLogitsLoss() (또는 BCELoss + Sigmoid),	CrossEntropyLoss() (Softmax 포함)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()  # Softmax 포함
optimizer = optim.Adam(model.parameters(), lr=LR)

# 4. 학습 및 시각화용 리스트
train_acc_list, val_acc_list = [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total, loss_total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)
    train_acc = correct / total
    train_acc_list.append(train_acc)

    # 검증
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in valid_loader:
            # Tensor 데이터를 지정된 디바이스(CPU 또는 GPU)로 이동시키고, 새 참조 리턴
            x, y = x.to(DEVICE), y.to(DEVICE)  
            outputs = model(x)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    val_acc = correct / total
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1} | Loss: {loss_total:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# 5. 학습 시각화
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()

# 6. 모델 저장
torch.save(model.state_dict(), "cat_dog_cnn.pth")

# 7. 모델 로드 (예시)
model.load_state_dict(torch.load("cat_dog_cnn.pth", map_location=DEVICE))
model.eval()

# 8. 실제 이미지 예측 함수
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    output = model(image_tensor)
    pred = output.argmax(1).item()
    plt.imshow(np.array(image))
    plt.title(f"Prediction: {class_names[pred]}")
    plt.axis('off')
    plt.show()

# 9. 예측 실행 예시
predict_image(data_path+'/val/cat/cat1.jpg')  # 실제 파일 경로 지정