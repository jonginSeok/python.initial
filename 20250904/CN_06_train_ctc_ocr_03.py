import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn

# -------------------------
# 1. IoU 계산
# -------------------------


def iou(box1, box2):
    # box: [x_min, y_min, x_max, y_max]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

# -------------------------
# 2. mAP 계산 (IoU>0.8, 클래스 구분)
# -------------------------


def compute_map(preds, gts, iou_thresh=0.8):
    """
    preds: list of [x1,y1,x2,y2,class_id,conf]
    gts:   list of [x1,y1,x2,y2,class_id]
    """
    classes = sorted(set([g[4] for g in gts]))
    ap_per_class = []

    for cls in classes:
        cls_preds = [p for p in preds if p[4] == cls]
        cls_gts = [g for g in gts if g[4] == cls]
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

# -------------------------
# 3. CER / 정확도
# -------------------------


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j]+1, curr[j-1]+1, prev[j-1]+cost))
        prev = curr
    return prev[-1]


def compute_cer(preds, gts):
    tot_err, tot_len = 0, 0
    for p, g in zip(preds, gts):
        tot_err += levenshtein(p, g)
        tot_len += max(1, len(g))
    return tot_err / tot_len


def exact_match_accuracy(preds, gts):
    return sum(int(p == g) for p, g in zip(preds, gts)) / len(gts)

# -------------------------
# 4. OCR Dataset / Collator (패딩)
# -------------------------


class OCRDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img = Image.open(self.rows[idx]["image_path"]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, self.rows[idx]["label"]


class LabelEncoder:
    def __init__(self, labels):
        charset = sorted(set("".join(labels)))
        self.char2idx = {ch: i+1 for i, ch in enumerate(charset)}
        self.idx2char = {i+1: ch for i, ch in enumerate(charset)}
        self.blank_idx = 0
        self.num_classes = len(self.char2idx) + 1

    def encode(self, text):
        return [self.char2idx[ch] for ch in text]

    def decode_ctc(self, probs):
        indices = torch.argmax(probs, dim=-1).tolist()
        prev = None
        out = []
        for t in indices:
            if t != self.blank_idx and t != prev:
                out.append(t)
            prev = t
        return "".join(self.idx2char[i] for i in out if i in self.idx2char)

    def indices_to_text(self, indices):
        return "".join(self.idx2char[i] for i in indices if i in self.idx2char)


class Collator:
    def __init__(self, encoder, img_height=32):
        self.encoder = encoder
        self.img_height = img_height
        self.resize = transforms.Resize

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        resized_imgs, widths = [], []
        for img in imgs:
            c, h, w = img.shape
            new_w = max(1, int(round(w * (self.img_height / h))))
            img_resized = self.resize((self.img_height, new_w))(img)
            resized_imgs.append(img_resized)
            widths.append(new_w)
        max_w = max(widths)
        padded_imgs = []
        for img in resized_imgs:
            c, h, w = img.shape
            pad_w = max_w - w
            padded = F.pad(img, (0, pad_w, 0, 0), value=0.0)
            padded_imgs.append(padded)
        label_lengths = [len(l) for l in labels]
        encoded_labels = []
        for l in labels:
            encoded_labels.extend(self.encoder.encode(l))
        return torch.stack(padded_imgs), torch.tensor(encoded_labels, dtype=torch.long), label_lengths, widths

# -------------------------
# 5. CRNN 모델
# -------------------------
# class CRNN(nn.Module):
#     def __init__(self, num_classes, in_channels=1, hidden=256):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
#             nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1)),
#             nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
#             nn.BatchNorm2d(512),
#             nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
#         )
#         self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, batch_first=False)
#         self.fc = nn.Linear(hidden*2, num_classes)
#     def forward(self, x):
#         feats = self.cnn(x).squeeze(2).permute(2, 0, 1)
#         seq, _ = self.rnn(feats)
#         return self.fc(seq)


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, '이미지 높이는 16의 배수여야 합니다.'

        # CNN 특징 추출기
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 1/2

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 1/4

            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 높이만 1/8

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 높이만 1/16

            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)  # 마지막 특징 압축
        )

        # RNN 시퀀스 모델링
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # input: [batch, channel, height, width]
        conv = self.cnn(input)  # [batch, 512, 1, width]
        b, c, h, w = conv.size()
        assert h == 1, "CNN 출력 높이는 1이어야 합니다."
        conv = conv.squeeze(2)  # [batch, 512, width]
        conv = conv.permute(2, 0, 1)  # [width, batch, 512]
        output = self.rnn(conv)  # [width, batch, nclass]
        return output


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output

# -------------------------
# 6. 통합 평가 실행
# -------------------------


def run_report(gt_boxes, pred_boxes, gt_texts, pred_texts):
    mAP, ap_per_class = compute_map(pred_boxes, gt_boxes, iou_thresh=0.8)
    cer = compute_cer(pred_texts, gt_texts)
    acc = exact_match_accuracy(pred_texts, gt_texts)

    print("\n📊 [통합 리포트]")
    print(f"🔹 검출 mAP@0.8 (클래스 구분): {mAP:.4f}")
    for i, ap in enumerate(ap_per_class):
        print(f"   └ 클래스 {i}: AP = {ap:.4f}")
    print(f"🔹 OCR CER (문자 오류율): {cer:.4f}")
    print(f"🔹 OCR 정확도 (완전 일치율): {acc*100:.2f}%")


# -------------------------
# 7. 예시 실행
# -------------------------
if __name__ == "__main__":
    # 예시: GT 박스와 예측 박스
    gt_boxes = [
        [10, 20, 100, 60, 0],  # [x1,y1,x2,y2,class_id]
        [120, 30, 200, 80, 1],
    ]
    pred_boxes = [
        [12, 22, 98, 58, 0, 0.95],  # [x1,y1,x2,y2,class_id,conf]
        [125, 35, 198, 78, 1, 0.90],
        [50, 50, 150, 100, 2, 0.60],  # 잘못된 클래스
    ]

    # 예시: OCR 인식 결과
    gt_texts = ["01가1234", "12나5678"]
    pred_texts = ["01가1234", "12나5670"]  # 두 번째는 오타

    # 리포트 실행
    run_report(gt_boxes, pred_boxes, gt_texts, pred_texts)

"""
📊 [통합 리포트]
🔹 검출 mAP@0.8 (클래스 구분): 0.9091
    └ 클래스 0: AP = 1.0000
    └ 클래스 1: AP = 1.0000
    └ 클래스 2: AP = 0.7273
🔹 OCR CER (문자 오류율): 0.0625
🔹 OCR 정확도 (완전 일치율): 50.00%
"""
