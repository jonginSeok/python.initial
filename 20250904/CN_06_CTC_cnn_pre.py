"""
🔠 2. CTC 기반 텍스트 인식 모델 구조 (PyTorch)
번호판 텍스트 인식을 위한 CTC 기반 모델은 일반적으로 CNN + BiLSTM + CTC Loss 구조를 사용합니다.

🧱 모델 구조 예시
"""
import torch
import torch.nn as nn

class CTCModel(nn.Module):
    def __init__(self, num_classes):
        super(CTCModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
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

"""
🧪 CTC Loss 적용
"""
ctc_loss = nn.CTCLoss(blank=0)
log_probs = nn.functional.log_softmax(output, dim=2)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

"""
🏷️ 3. 라벨 자동 생성 코드 (번호판 crop 이미지 기준)
Roboflow 또는 YOLO 추론 결과에서 바운딩 박스를 받아 텍스트 라벨을 자동 생성하는 코드입니다.

📄 예시 코드
"""

import os
import json

def generate_labels(image_folder, label_dict, save_path):
    """
    image_folder: cropped 이미지가 저장된 폴더
    label_dict: {'origin_name_plate.jpg': '12가3456', ...}
    save_path: 라벨 저장 경로
    """
    os.makedirs(save_path, exist_ok=True)
    for img_name, text in label_dict.items():
        label_file = os.path.join(save_path, img_name.replace('.jpg', '.txt'))
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write(text)

# 예시 딕셔너리
label_dict = {
    "car123_plate.jpg": "12가3456",
    "car123_s1.jpg": "서울",
    "car123_d1.jpg": "12가",
    "car123_d2.jpg": "3456"
}

generate_labels("cropped_images/", label_dict, "labels/")
