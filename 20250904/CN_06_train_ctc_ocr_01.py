import os
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


# =========================
# 1. CSV 읽기
# =========================
def load_csv_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV 파일 없음: {csv_path}")
        return rows

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
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
            img = self.transform(img)
        return img, row["label"]


# =========================
# 3. 라벨 인코더
# =========================
class LabelEncoder:
    def __init__(self, labels):
        charset = sorted(set("".join(labels)))
        self.char2idx = {ch: i + 1 for i, ch in enumerate(charset)}  # 0은 blank
        self.idx2char = {i + 1: ch for i, ch in enumerate(charset)}
        self.blank_idx = 0

    def encode(self, text):
        return [self.char2idx[ch] for ch in text]

    def decode(self, indices):
        return "".join([self.idx2char[i] for i in indices if i != self.blank_idx])


# =========================
# 4. Collate (비율 유지 + 패딩)
# =========================
class Collator:
    def __init__(self, encoder, img_height=32):
        self.encoder = encoder
        self.img_height = img_height

    def __call__(self, batch):
        imgs, labels = zip(*batch)

        resized_imgs = []
        widths = []
        for img in imgs:
            c, h, w = img.shape
            new_w = int(w * (self.img_height / h))
            img_resized = transforms.Resize((self.img_height, new_w))(img)
            resized_imgs.append(img_resized)
            widths.append(new_w)

        max_w = max(widths)
        padded_imgs = []
        for img in resized_imgs:
            c, h, w = img.shape
            pad_w = max_w - w
            padded = F.pad(img, (0, pad_w, 0, 0), value=0)
            padded_imgs.append(padded)

        label_lengths = [len(l) for l in labels]
        encoded_labels = []
        for l in labels:
            encoded_labels.extend(self.encoder.encode(l))

        return (
            torch.stack(padded_imgs),
            torch.tensor(encoded_labels, dtype=torch.long),
            label_lengths,
            widths,
        )


# =========================
# 5. 실행 예시
# =========================
if __name__ == "__main__":
    csv_path = r"C:\Users\ngins\Git\python.initial\20250904\runs\cropped_images_labels.csv"

    rows = load_csv_rows(csv_path)
    if not rows:
        exit("[ERROR] 학습할 데이터가 없습니다.")

    train_rows, val_rows = train_test_split(rows, test_size=0.2, random_state=42)
    encoder = LabelEncoder([r["label"] for r in rows])

    transform = transforms.ToTensor()
    train_ds = OCRDataset(train_rows, transform=transform)
    val_ds = OCRDataset(val_rows, transform=transform)

    collator = Collator(encoder, img_height=32)

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collator
    )

    # 테스트
    for imgs, labels, label_lengths, widths in train_loader:
        print(f"이미지 배치 크기: {imgs.shape}")
        print(f"인코딩 라벨: {labels}")
        print(f"라벨 길이: {label_lengths}")
        print(f"이미지 폭: {widths}")
        break
