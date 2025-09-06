import os
import re
import csv

# 경로 설정
target_dir = r"20250904/runs/cropped_images"
output_csv = r"20250904/runs/cropped_images_labels.csv"

IMG_EXTS = (".jpg", ".jpeg", ".png")
rows = []

for fname in os.listdir(target_dir):
    if not fname.lower().endswith(IMG_EXTS):
        continue

    name, ext = os.path.splitext(fname)
    parts = name.split("_")
    if len(parts) < 2:
        continue

    prefix = parts[0]  # 예: 01가1134-2
    suffix = parts[-1] # plate, s1, d1, d2

    label = None
    m = re.match(r"^(\d+)([가-힣])(\d+)(?:-\d+)?$", prefix)
    if not m:
        print(f"⚠ 패턴 불일치: {fname}")
        continue

    num1, hangul, num2 = m.groups()

    if suffix in ("plate", "s1"):
        label = f"{num1}{hangul}{num2}"
    elif suffix == "d1":
        label = f"{num1}{hangul}"
    elif suffix == "d2":
        label = num2

    if label:
        rows.append({
            "image_path": os.path.abspath(os.path.join(target_dir, fname)).replace("\\", "/"),
            "label": label
        })

# utf-8-sig는 UTF-8에 BOM(Byte Order Mark)을 붙여서, Windows Excel/메모장에서도 자동으로 UTF-8로 인식하게 해줍니다.
with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
    writer.writeheader()
    writer.writerows(rows)

# 한글깨짐.
# with open(output_csv, "w", newline="", encoding="utf-8") as f:
#     writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
#     writer.writeheader()
#     writer.writerows(rows)

print(f"✅ CSV 저장 완료: {output_csv}")


"""
해결 방법

---------------------
1. CSV 저장 시 BOM 제거
---------------------
CSV 저장할 때 encoding="utf-8"로 저장하면 BOM이 안 붙습니다.

python
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
    writer.writeheader()
    writer.writerows(rows)

---------------------
2. CSV 읽을 때 BOM 제거
---------------------
이미 저장된 CSV를 읽을 때는 encoding="utf-8-sig"로 열면 BOM이 자동 제거됩니다.

python
with open(cfg.csv_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    rows = list(reader)

---------------------
3. 헤더 스킵
---------------------
image_path,label 같은 헤더가 데이터로 들어가지 않게 첫 줄은 건너뛰세요.

python
with open(cfg.csv_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 건너뛰기
    rows = list(reader)

---------------------
추천 조합
---------------------
CSV 저장: utf-8 (BOM 없음)

CSV 읽기: utf-8 + next(reader)로 헤더 스킵
"""