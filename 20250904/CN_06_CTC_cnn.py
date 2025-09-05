"""
🧪 1. CTC 학습 루프 전체 코드 (PyTorch 기반)
이 코드는 CNN + BiLSTM + CTC Loss 구조의 모델을 학습하는 전체 루프입니다.

📌 주요 구성
이미지: 번호판 crop 이미지 (흑백 또는 RGB → 흑백 변환)

라벨: 텍스트 (예: "12가3456")

CTC Loss 사용

라벨 인코딩: 문자 → 인덱스

🧱 학습 루프 예시

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# 문자 사전 정의
alphabet = "0123456789가나다라마바사아자차카타파하"
char_to_idx = {char: idx + 1 for idx, char in enumerate(alphabet)}  # 0은 blank
idx_to_char = {idx: char for char, idx in char_to_idx.items()}


# 라벨 인코딩 함수
def encode_label(text):
    return [char_to_idx[c] for c in text]


class CTCModel(nn.Module):
    def __init__(self, num_classes):
        super(CTCModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.rnn = nn.LSTM(128, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)  # 256*2 (bi-directional)

    def forward(self, x):
        x = self.cnn(x)  # [B, C, H, W]
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.view(x.size(0), x.size(1), -1)  # [B, W, C*H]
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


# 데이터셋 클래스
class PlateDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.open(self.image_paths[idx]).convert("L")
        img = self.transform(img)
        label = torch.tensor(encode_label(self.labels[idx]), dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.image_paths)


# 모델 정의 (이전 답변의 CTCModel 사용)
model = CTCModel(num_classes=len(alphabet) + 1).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# 데이터 준비
transform = transforms.Compose(
    [
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = PlateDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)

# 학습 루프
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        images, targets = zip(*batch)
        images = torch.stack(images).to("cuda")
        targets = [t.to("cuda") for t in targets]

        # 입력 길이와 타겟 길이 계산
        input_lengths = torch.full(
            size=(len(images),), fill_value=images.size(3) // 4, dtype=torch.long
        )
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

        targets_concat = torch.cat(targets)
        outputs = model(images)  # [B, T, C]
        log_probs = nn.functional.log_softmax(outputs, dim=2)

        loss = ctc_loss(
            log_probs.permute(1, 0, 2), targets_concat, input_lengths, target_lengths
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")


"""
🧬 2. 데이터 증강 (번호판 이미지에 적합한 방식)
번호판 인식에 적합한 증강은 왜곡 방지와 조명/노이즈 다양성 확보가 핵심입니다.

🎨 추천 증강 기법
"""

from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((32, 128)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5
        ),
        transforms.RandomRotation(degrees=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

"""
🔍 3. 추론 결과 시각화 코드
CTC 모델의 출력은 시퀀스 형태이므로 디코딩이 필요합니다. 아래는 추론 후 결과를 이미지에 시각화하는 코드입니다.

📸 디코딩 함수

"""


def decode_output(output):
    output = output.argmax(dim=2)  # [B, T]
    texts = []
    for seq in output:
        prev = -1
        text = ""
        for idx in seq:
            if idx != prev and idx != 0:
                text += idx_to_char[idx.item()]
            prev = idx
        texts.append(text)
    return texts


"""
🖼️ 시각화 예시
"""

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    for img_path in test_image_paths:
        img = Image.open(img_path).convert("L")
        input_img = transform(img).unsqueeze(0).to("cuda")
        output = model(input_img)
        pred_text = decode_output(output)[0]

        plt.imshow(img, cmap="gray")
        plt.title(f"Predicted: {pred_text}")
        plt.axis("off")
        plt.show()


"""
🚀 다음 단계
이제 전체 파이프라인이 거의 완성됐어요. 원하시면:

모델 저장 및 추론 API 구성

ONNX 또는 TensorRT로 경량화

실시간 카메라 스트림 적용

"""
