# Dataset 클래스
from torch.utils.data import Dataset
from PIL import Image, ImageOps
# pip install torch
# pip install pillow
# pip install matplotlab

class CarrotDatasetWithSize(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.samples = []           # 파일 경로와 라벨을 짝지어 리스트에 저장
        self.transform = transform
        self.label_map = label_map

        # samples 리스트에 이미지파일 경로와 라벨을 모두 저장
        for class_name, label in label_map.items():   # {'cat': 0, 'dog': 1}
            class_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(class_path, fname), label))  # (이미지 파일 경로, 0/1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # 원본 이미지 크기 정보
        orig_w, orig_h = image.size
        area = orig_w * orig_h
        aspect_ratio = orig_w / orig_h
        size_features = torch.tensor([orig_w, orig_h, area, aspect_ratio], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, size_features, label   # 이미지, 크기, 라벨 리턴


# CNN
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# 신경망을 정의할 때는 Batch 크기를 고려하지 않지만 Tensor 연산은 배치단위 병렬처리가 기본임
# 신경망에 데이터를 전달할 때는 Batch 단위로 전달해야 하며 신경망 출력측에서도 배치단위로 출력된다
# 개발자는 하나의 샘플을 기준으로 신경망 구조만 정의하면 되며,
# 여러 샘플을 병렬로 처리하는 역할은 프레임워크(Pytorch, TensorFlow)가 자동으로 수행한다.
class CarrotCNNWithSize(nn.Module):
    def __init__(self):
        super().__init__()    # 신경망을 선언할 때는 Batch 크기를 고려하지 않으며 Tensor연산은 배치단위 병렬처리를 지원함
        self.conv = nn.Sequential(  # 3channel, 16filters, filter-size 3
            nn.Conv2d(3, 16, 3, padding=1),  # 224x224x3 -> 224x224x16, padding=1은 1픽셀 추가하여 출력크기 유지
            nn.ReLU(),
            nn.MaxPool2d(2),        # -> 112x112x16, 이미지 크기를 1/2로 축소(국소적 특징 요약)
            nn.Conv2d(16, 32, 3, padding=1), # 112x112x16 -> 112x112x32
            nn.ReLU(),
            nn.MaxPool2d(2),     # -> 56x56x32
        )
        self.flat_size = 56 * 56 * 32
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size + 4, 128),  # +4 for size info
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, size_feats):    # 순전파(forward) 정의
        x = self.conv(x)                        # x: (B, 56, 56, 32)
        x = x.view(x.size(0), -1)  # Flatten    # x: (B, 56*56*32)  1차원 데이터가 Batch 만큼 리턴됨. -1은 자동으로 설정
        x = torch.cat([x, size_feats], dim=1)   # 각 배치에 이미지 사이즈 정보 추가. 2번째 차원에 추가
        return self.fc(x)


# Letterbox 클래스 선언
from PIL import ImageOps

class Letterbox:
    def __init__(self, size, color=(128, 128, 128)): # LetterBox색 지정
        self.size = size  # 정사각형 대상 사이즈 (ex: 224)
        self.color = color  # 패딩 색 (회색)

    def __call__(self, img):
        # 원본 크기
        iw, ih = img.size
        scale = min(self.size / iw, self.size / ih)   # 이미지 폭,높이 중 큰 것과 CNN입력 크기의 비 # scale :분모
        nw, nh = int(iw * scale), int(ih * scale)     # 이미지 폭,높이 중 큰 것을 CNN입력 크기에 맞춘다
        # 리사이즈
        img = img.resize((nw, nh), Image.BILINEAR)    # 이미지 크기를 CNN입력 크기로 변경 # BILINEAR:양선보강법
        # Image.BILINEAR:속도와 품질의 균형이 좋은 보간법. 특히 딥러닝용 이미지 전처리 등에서 자주 사용
        # 패딩
        new_img = Image.new("RGB", (self.size, self.size), self.color)  # CNN입력 크기의 빈 이미지 생성
        pad_left = (self.size - nw) // 2              # 좌우 여백의 크기 # //: 정수 반환
        pad_top = (self.size - nh) // 2               # 상하 여백의 크기 # 중간에 맞추기 위해
        new_img.paste(img, (pad_left, pad_top))       # 빈 이미지에 실제 이미지를 붙여넣기
        return new_img
    

# Dataset 에 전달할 transform 생성(수동으로도 호출 가능함)
transform = transforms.Compose([
    Letterbox(224),                     # 이미지 비율 유지
    #transforms.Resize((224, 224)),      # 이미지를 고정 크기로 설정
    transforms.ToTensor(),              # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize([0.5]*3, [0.5]*3)  # 빠르고 안정적인 학습을 위한 정규화(0~1 -> -1~1), (x-0.5)/0.5
])

# data_path = "/content/drive/MyDrive/Python_AI/YOLO/Carrot Classification/CARROT원본"
data_path = "C:\\Users\\ngins\\Git\\python.initial\\dataset\\dental\\images"
BATCH_SIZE = 4
label_map = {'BAD': 0, 'GOOD': 1}  # class_names = ['BAD', 'GOOD']
train_dataset = CarrotDatasetWithSize(data_path + '\\train', label_map, transform=transform)
valid_dataset = CarrotDatasetWithSize(data_path + '\\val', label_map, transform=transform)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = list(label_map.keys())  # ['BAD', 'GOOD']
# class_names    

# 학습 루프
BATCH_SIZE = 4
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE=", DEVICE)

model = CarrotCNNWithSize().to(DEVICE)   # 모델 생성 및 GPU에 이동
criterion = nn.CrossEntropyLoss()        # Softmax 포함
optimizer = optim.Adam(model.parameters(), lr=LR)

train_acc_list, val_acc_list = [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total, loss_total = 0, 0, 0
    for x, size_feats, y in train_loader:   # Batch 단위 로드
        x, y, size_feats = x.to(DEVICE), y.to(DEVICE), size_feats.to(DEVICE)   # image, label, size_feats GPU에 이동
        optimizer.zero_grad()
        outputs = model(x, size_feats)   # __call__() -> forward()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)    # 라벨의 Batch 샘플 수. 총 샘플 수 합산
    train_acc = correct / total
    train_acc_list.append(train_acc)   # epoch 정확도 리스트에 저장

    # 검증
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, size_feats, y in valid_loader:
            x, y, size_feats = x.to(DEVICE), y.to(DEVICE), size_feats.to(DEVICE)   #
            outputs = model(x, size_feats)   # GPU에서 작동하는 모델이 리턴하는 값과 연산하는 대상 데이터도 GPU에 존재해야 한다
            correct += (outputs.argmax(1) == y).sum().item()   # GPU에서 리턴된 값과 연산하는 대상 데이터 y
            total += y.size(0)
    val_acc = correct / total
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1} | Loss: {loss_total:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
