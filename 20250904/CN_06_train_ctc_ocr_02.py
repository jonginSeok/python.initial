import os
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from typing import List, Tuple
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================
# 1. CSV 읽기
# =========================
def load_csv_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV 파일 없음: {csv_path}")
        return rows

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row["image_path"].strip()
            label = row["label"].strip()
            if not os.path.exists(image_path):
                print(f"[경고] 파일 없음: {image_path}")
                continue
            rows.append({"image_path": image_path, "label": label})

    print(f"[INFO] CSV 로드 완료: {len(rows)}개 항목")
    return rows

# =========================
# 2. Dataset
# =========================
class OCRDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["image_path"]).convert("L")  # 흑백
        if self.transform:
            img = self.transform(img)  # (1, H, W)
        return img, row["label"]

# =========================
# 3. 라벨 인코더 (CTC용)
# =========================
class LabelEncoder:
    def __init__(self, labels: List[str]):
        charset = sorted(set("".join(labels)))
        self.char2idx = {ch: i+1 for i, ch in enumerate(charset)}  # 0은 blank
        self.idx2char = {i+1: ch for i, ch in enumerate(charset)}
        self.blank_idx = 0
        self.num_classes = len(self.char2idx) + 1  # + blank

    def encode(self, text: str) -> List[int]:
        return [self.char2idx[ch] for ch in text]

    def decode_ctc(self, logits: torch.Tensor) -> str:
        # logits: (T, C) after softmax/argmax step handling here (we’ll pass probs)
        # Greedy: argmax per timestep, collapse repeats, remove blank
        indices = torch.argmax(logits, dim=-1).tolist()
        prev = None
        out = []
        for t in indices:
            if t != self.blank_idx and t != prev:
                out.append(t)
            prev = t
        # map to chars
        return "".join(self.idx2char[i] for i in out if i in self.idx2char)

    def indices_to_text(self, indices: List[int]) -> str:
        return "".join(self.idx2char[i] for i in indices if i in self.idx2char)

# =========================
# 4. Collate (비율 유지 + 패딩)
# =========================
class Collator:
    def __init__(self, encoder: LabelEncoder, img_height=32):
        self.encoder = encoder
        self.img_height = img_height
        self.resize = transforms.Resize  # to keep torchvision op creation inside __call__

    def __call__(self, batch):
        imgs, labels = zip(*batch)

        # 1) 높이 정규화(고정), 폭은 비율 유지
        resized_imgs = []
        widths = []
        for img in imgs:
            c, h, w = img.shape
            new_w = max(1, int(round(w * (self.img_height / h))))
            img_resized = self.resize((self.img_height, new_w))(img)
            resized_imgs.append(img_resized)
            widths.append(new_w)

        # 2) 동일 폭으로 패딩 (오른쪽 제로패드)
        max_w = max(widths)
        padded_imgs = []
        for img in resized_imgs:
            c, h, w = img.shape
            pad_w = max_w - w
            padded = F.pad(img, (0, pad_w, 0, 0), value=0.0)
            padded_imgs.append(padded)

        # 3) 라벨 인코딩 (CTC target은 1D concat + lengths 필요)
        label_lengths = [len(l) for l in labels]
        encoded_labels = []
        for l in labels:
            encoded_labels.extend(self.encoder.encode(l))

        batch_imgs = torch.stack(padded_imgs)  # (B, 1, H, Wmax)
        batch_labels = torch.tensor(encoded_labels, dtype=torch.long)  # (sum_T,)
        return batch_imgs, batch_labels, label_lengths, widths  # widths는 원본(패딩 전) 새 폭

# =========================
# 5. CRNN 모델 (H=32 가정, 가로 방향 시퀀스)
# =========================
class CRNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, hidden: int = 256):
        super().__init__()
        # CNN: H를 1로 압축, W는 4배 다운샘플 (대략)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),  # (B,64,H,W)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                   # (B,64,H/2,W/2)

            nn.Conv2d(64, 128, 3, 1, 1),          # (B,128,H/2,W/2)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                   # (B,128,H/4,W/4)

            # height 방향을 1로 squeeze하기 위해 커널/스트라이드 조정
            nn.Conv2d(128, 256, 3, 1, 1),         # (B,256,H/4,W/4)
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(self._pool_h(32), 1), stride=(self._pool_h(32), 1)),  # H -> 1, W 유지

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
        )
        # BiLSTM over width time-steps
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=False
        )
        self.fc = nn.Linear(hidden * 2, num_classes)  # bidirectional

    def _pool_h(self, input_h: int) -> int:
        # 입력 높이를 32로 맞췄다고 가정. 위에서 두 번 MaxPool2d(2,2) → H = input_h / 4
        # 여기서 그 H를 한 번에 1로 줄이기 위해 커널=H, 스트라이드=H
        return max(1, input_h // 4)

    def forward(self, x):  # x: (B,1,H,W)
        feats = self.cnn(x)         # (B, 512, 1, W')
        feats = feats.squeeze(2)    # (B, 512, W')
        feats = feats.permute(2, 0, 1)  # (T=W', B, 512)
        seq, _ = self.rnn(feats)    # (T, B, 2*hidden)
        logits = self.fc(seq)       # (T, B, C)
        return logits

# =========================
# 6. 유틸: CTC 길이, 디코딩, CER
# =========================
def ctc_input_lengths(widths: List[int]) -> List[int]:
    # CNN에서 W가 대략 4배 다운샘플 (두 번의 2x2 풀링)
    # 이후 H squeeze는 W에 영향 없음
    return [max(1, w // 4) for w in widths]

def greedy_decode_batch(logits: torch.Tensor, encoder: LabelEncoder) -> List[str]:
    # logits: (T, B, C)
    probs = logits.softmax(dim=-1)          # (T, B, C)
    preds = probs.detach().cpu()            # 안전
    out_texts = []
    T, B, C = preds.shape
    for b in range(B):
        path = preds[:, b, :]               # (T, C)
        text = encoder.decode_ctc(path)
        out_texts.append(text)
    return out_texts

def levenshtein(a: str, b: str) -> int:
    # 편집거리
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,      # deletion
                curr[j-1] + 1,    # insertion
                prev[j-1] + cost  # substitution
            ))
        prev = curr
    return prev[-1]

def compute_cer(preds: List[str], gts: List[str]) -> float:
    tot_err = 0
    tot_len = 0
    for p, g in zip(preds, gts):
        tot_err += levenshtein(p, g)
        tot_len += max(1, len(g))
    return tot_err / tot_len

def exact_match_accuracy(preds: List[str], gts: List[str]) -> float:
    hit = sum(int(p == g) for p, g in zip(preds, gts))
    return hit / len(gts)

# =========================
# 7. 학습/검증 루프
# =========================
def train_one_epoch(model, loader, optimizer, criterion, encoder, device):
    model.train()
    total_loss = 0.0
    for imgs, labels, label_lengths, widths in loader:
        imgs = imgs.to(device)  # (B,1,H,W)
        labels = labels.to(device)  # (sum_T,)

        optimizer.zero_grad()
        logits = model(imgs)                 # (T,B,C)
        input_lengths = torch.tensor(ctc_input_lengths(widths), dtype=torch.long, device=device)  # (B,)
        target_lengths = torch.tensor(label_lengths, dtype=torch.long, device=device)  # (B,)

        # CTC expects log-probs
        log_probs = logits.log_softmax(dim=-1)  # (T,B,C)
        loss = criterion(log_probs, labels, input_lengths, target_lengths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, criterion, encoder, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_gts = []
    for imgs, labels, label_lengths, widths in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)  # (T,B,C)

        input_lengths = torch.tensor(ctc_input_lengths(widths), dtype=torch.long, device=device)
        target_lengths = torch.tensor(label_lengths, dtype=torch.long, device=device)
        log_probs = logits.log_softmax(dim=-1)

        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        total_loss += loss.item()

        # 디코딩
        pred_texts = greedy_decode_batch(logits, encoder)
        # GT 복원
        gt_texts = []
        offset = 0
        for L in label_lengths:
            indices = labels.cpu().tolist()[offset:offset+L]
            gt_texts.append(encoder.indices_to_text(indices))
            offset += L

        all_preds.extend(pred_texts)
        all_gts.extend(gt_texts)

    cer = compute_cer(all_preds, all_gts)
    acc = exact_match_accuracy(all_preds, all_gts)
    return total_loss / max(1, len(loader)), cer, acc

# =========================
# 8. 메인 실행
# =========================
def main():
    # 경로 수정
    csv_path = r"C:\Users\ngins\Git\python.initial\20250904\runs\cropped_images_labels_output.csv"

    rows = load_csv_rows(csv_path)
    if not rows:
        raise SystemExit("[ERROR] 학습할 데이터가 없습니다.")

    train_rows, val_rows = train_test_split(rows, test_size=0.1, random_state=42, shuffle=True)

    encoder = LabelEncoder([r["label"] for r in rows])

    # transform = transforms.ToTensor()
    # 이미지 전처리 (고정 크기)
    transform = transforms.Compose([
        transforms.Resize((32, 128)),  # (높이, 폭)
        transforms.ToTensor()
    ])

    BATCH_SIZE = 4 #32

    train_ds = OCRDataset(train_rows, transform=transform)
    val_ds = OCRDataset(val_rows, transform=transform)

    collator = Collator(encoder, img_height=32)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collator)

    # 테스트
    for imgs, labels, label_lengths, widths in train_loader:
        print(f"이미지 배치 크기: {imgs.shape}")
        print(f"인코딩 라벨: {labels}")
        print(f"라벨 길이: {label_lengths}")
        print(f"이미지 폭: {widths}")
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=encoder.num_classes, in_channels=1, hidden=256).to(device)

    criterion = nn.CTCLoss(blank=encoder.blank_idx, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2) # , verbose=True

    epochs = 20
    best_cer = float("inf")

    print(f"[INFO] 학습 시작: epochs={epochs}, device={device}, classes={encoder.num_classes}")

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, encoder, device)
        val_loss, val_cer, val_acc = evaluate(model, val_loader, criterion, encoder, device)
        scheduler.step(val_loss)

        print(f"[Epoch:{epoch:02d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | CER={val_cer:.4f} | Acc={val_acc*100:.2f}%")

        # 체크포인트
        if val_cer < best_cer:
            best_cer = val_cer
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "encoder": {
                    "char2idx": encoder.char2idx,
                    "idx2char": encoder.idx2char,
                    "blank_idx": encoder.blank_idx
                }
            }
            os.makedirs("20250904/runs/checkpoints", exist_ok=True)
            torch.save(ckpt, f"20250904/runs/checkpoints/crnn_ctc_best.pth")
            print(f"[INFO] Checkpoint saved. (best CER={best_cer:.4f})")

    # 마지막 평가(베스트 기준)
    print(f"[DONE] Best CER={best_cer:.4f}")

if __name__ == "__main__":
    main()
