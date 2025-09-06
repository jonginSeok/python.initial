import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
from collections import OrderedDict
import string

# ===== 1. 데이터셋 로드 =====
csv_path = "20250904/runs/cropped_images_labels.csv"
df = pd.read_csv(csv_path)

# 1) 기본 보강 세트
digits = list("0123456789")
latin_upper = list(string.ascii_uppercase)
sep_tokens = [" ", "-", "·", "."]
korean_base = list("가나다라마바사아자차카타파하")
korean_extra = list("허호배영외군임학관준전렌택공")

base_charset = OrderedDict()
for c in digits + korean_base + korean_extra + sep_tokens + latin_upper:
    base_charset[c] = True

# 2) CSV 스캔: 실제 등장 문자 수집
csv_charset = OrderedDict()
for text in df["label"].astype(str).tolist():
    for ch in text:
        csv_charset[ch] = True

# 3) 병합: CSV에서 등장한 문자를 우선 포함하고, 기본 보강은 필요한 것만 추가
merged_charset = OrderedDict()
for ch in csv_charset.keys():
    merged_charset[ch] = True
for ch in base_charset.keys():
    if ch not in merged_charset:
        merged_charset[ch] = True

# 4) 최종 charset (CTC blank=0 예약 → 실제 문자는 1부터 시작)
charset_list = list(merged_charset.keys())

print(f"charset_list : {charset_list}")

# 문자 집합 정의 (번호판에 등장할 수 있는 문자)
# charset = "0123456789가나다라마바사아자차카타파하"
char2idx = {c: i+1 for i, c in enumerate(charset_list)}  # 0은 CTC blank
idx2char = {i+1: c for i, c in enumerate(charset_list)}

# ===== 2. Dataset 클래스 =====
class OCRDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label_str = self.df.iloc[idx]['label']
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label_idx = [char2idx[c] for c in label_str]
        return image, torch.tensor(label_idx, dtype=torch.long), label_str

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor()
])

dataset = OCRDataset(df, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda b: collate_fn(b))

# ===== 3. Collate 함수 =====
def collate_fn(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels])
    labels = torch.cat(labels)
    return images, labels, label_lengths, texts

# ===== 4. CRNN 모델 정의 =====
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128*8, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # [B, C, H, W]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c*h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# ===== 5. 학습 준비 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(num_classes=len(charset_list)+1).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ===== 6. 학습 루프 =====
for epoch in range(50):
    model.train()
    total_loss = 0
    for images, labels, label_lengths, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # [B, W, num_classes]
        outputs = outputs.log_softmax(2)
        input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(1), dtype=torch.long)
        loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[E{epoch+1}] Loss: {total_loss/len(loader):.4f}")

# ===== 7. OCR 평가 (CER, Accuracy) =====
def decode(preds):
    pred_texts = []
    for p in preds:
        p = torch.argmax(p, dim=1).cpu().numpy()
        prev = -1
        text = ""
        for idx in p:
            if idx != prev and idx != 0:
                text += idx2char[idx]
            prev = idx
        pred_texts.append(text)
    return pred_texts

def levenshtein(a: str, b: str) -> int:
    # """문자열 a와 b의 레벤슈타인 거리(편집 거리)를 계산"""
    if len(a) < len(b):
        a, b = b, a
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr_row = [i]
        for j, cb in enumerate(b, 1):
            insertions = prev_row[j] + 1
            deletions = curr_row[j - 1] + 1
            substitutions = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

model.eval()
all_gt, all_pred = [], []
with torch.no_grad():
    for images, _, _, texts in loader:
        images = images.to(device)
        outputs = model(images).log_softmax(2)
        preds = decode(outputs)
        all_gt.extend(texts)
        all_pred.extend(preds)

acc = accuracy_score(all_gt, all_pred)
cer = np.mean([levenshtein(gt, pr)/len(gt) for gt, pr in zip(all_gt, all_pred)])
print(f"OCR Accuracy: {acc*100:.2f}% | CER: {cer:.4f}")

# ===== 8. mAP@0.8 계산 (예시) =====
# GT 박스와 예측 박스가 있다고 가정하고 IoU 계산 후 mAP 산출
# (여기서는 함수 틀만 제공)
# def compute_map(gt_boxes, pred_boxes, iou_thresh=0.8):
    # gt_boxes, pred_boxes: {image_id: [(x1,y1,x2,y2,class), ...]}
    # IoU 계산 후 AP → mAP
    # pass
def compute_map(gt_boxes, pred_boxes, iou_thresh=0.8):
    # """
    # preds: list of [x1,y1,x2,y2,class_id,conf]
    # gts:   list of [x1,y1,x2,y2,class_id]
    # """
    classes = sorted(set([g[4] for g in gts]))
    ap_per_class = []

    for cls in classes:
        cls_preds = [p for p in gt_boxes if p[4] == cls]
        cls_gts = [g for g in pred_boxes if g[4] == cls]
        n_gt = len(cls_gts)

        cls_preds.sort(key=lambda x: x[5], reverse=True)
        matched = set()
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, pred in enumerate(cls_preds):
            best_iou = 0
            best_gt_idx = -1
            for j, gt in enumerate(cls_gts):
                if j in matched:
                    continue
                iou_val = iou(pred, gt)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = j
            if best_iou >= iou_thresh:
                matched.add(best_gt_idx)
                tp[i] = 1
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / (n_gt + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap = 0
        for t in np.linspace(0, 1, 11):
            p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0
            ap += p
        ap /= 11
        ap_per_class.append(ap)

    return np.mean(ap_per_class), ap_per_class

# ===== 9. 최종 리포트 =====
print("=== Final Report ===")
print(f"OCR CER: {cer:.4f}")
print(f"OCR Accuracy: {acc*100:.2f}%")
print(f"Detection mAP@0.8: {0.9124:.4f} (예시)")
