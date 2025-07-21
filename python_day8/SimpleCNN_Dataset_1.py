import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from SimpleCNN_Dataset_0 import CustomImageDataset

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
# data_path = "/content/drive/MyDrive/Python_AI/CNN/dataset"
data_path = "./dataset"
#train_dataset = datasets.ImageFolder(root=data_path+'/train', transform=transform)
#valid_dataset = datasets.ImageFolder(root=data_path+'/val', transform=transform)

# label_map 정의
label_map = {'cat': 0, 'dog': 1}
class_names = list(label_map.keys())
 
# 커스텀 Dataset 적용
train_dataset = CustomImageDataset(root_dir=data_path+'/train', label_map=label_map, transform=transform)
valid_dataset = CustomImageDataset(root_dir=data_path+'/val', label_map=label_map, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 모델이 순서에 영향을 받지 않도록 매 epoch마다 무작위로 섞는다
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False) # 데이터 순서 고정

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