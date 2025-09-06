import os
import cv2
import numpy as np

# 경로 설정
images_dir = "20250904/CarNumber.v4i.yolov8-obb/train/images"
labels_dir = "20250904/CarNumber.v4i.yolov8-obb/train/labels"
output_dir = "20250904/runs/cropped_images"
os.makedirs(output_dir, exist_ok=True)

# 클래스 ID (필요시 수정)
LICENSE_PLATE_CLS = 0
TEXT_CLS = 1
IMAGE_EXT = ".jpg"

def yolo_to_pixel_coords(coords, img_w, img_h):
    pts = []
    for i in range(0, len(coords), 2):
        x = float(coords[i]) * img_w
        y = float(coords[i+1]) * img_h
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.int32)

def clamp_bbox(x, y, w, h, img_w, img_h):
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(0, min(w, img_w - x))
    h = max(0, min(h, img_h - y))
    return x, y, w, h

for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    # 원래 base_name
    base_name = os.path.splitext(label_file)[0]

    # _jpg.rf 이후 부분 제거
    if "_jpg.rf" in base_name:
        base_name = base_name.split("_jpg.rf")[0]

    image_path = os.path.join(images_dir, os.path.splitext(label_file)[0] + IMAGE_EXT)
    label_path = os.path.join(labels_dir, label_file)

    if not os.path.exists(image_path):
        print(f"이미지 없음: {image_path}")
        continue

    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    plate_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    text_items = []  # (y_center, mask, polygon)

    for line in lines:
        parts = line.split()
        cls = int(parts[0])
        coords = parts[1:]
        polygon = yolo_to_pixel_coords(coords, img_w, img_h)
        if len(polygon) < 3:
            continue

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        if cls == LICENSE_PLATE_CLS:
            plate_mask = cv2.bitwise_or(plate_mask, mask)

        elif cls == TEXT_CLS:
            x, y, w, h = cv2.boundingRect(polygon)
            x, y, w, h = clamp_bbox(x, y, w, h, img_w, img_h)
            if w > 0 and h > 0:
                y_center = y + h / 2
                text_items.append((y_center, mask, polygon))

    # 번호판 전체 저장
    if np.any(plate_mask):
        plate_img = cv2.bitwise_and(img, img, mask=plate_mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_plate{IMAGE_EXT}"), plate_img)

    # 텍스트 개수에 따른 저장
    if len(text_items) == 1:
        # _s1 저장
        _, m, p = text_items[0]
        masked = cv2.bitwise_and(img, img, mask=m)
        x, y, w, h = cv2.boundingRect(p)
        x, y, w, h = clamp_bbox(x, y, w, h, img_w, img_h)
        crop = masked[y:y+h, x:x+w]
        if crop.size > 0:
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_s1{IMAGE_EXT}"), crop)

    elif len(text_items) == 2:
        # _d1, _d2 저장
        text_items.sort(key=lambda t: t[0])  # y_center 기준 정렬
        for idx, tag in zip([0, 1], ["d1", "d2"]):
            _, m, p = text_items[idx]
            masked = cv2.bitwise_and(img, img, mask=m)
            x, y, w, h = cv2.boundingRect(p)
            x, y, w, h = clamp_bbox(x, y, w, h, img_w, img_h)
            crop = masked[y:y+h, x:x+w]
            if crop.size > 0:
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_{tag}{IMAGE_EXT}"), crop)

print("✅ 처리 완료1")

import os
import re
#from collections import defaultdict
import itertools



# 경로 설정
target_dir = r"D:\Users\ngins\Projects\2025.07.10_심화과정-인공지능_YOLO기반_부트캠프\Datasets\[원천]자동차번호판OCR데이터"  # 원본 폴더
source_dir = r"C:\Users\ngins\Git\python.initial\20250904\runs\cropped_images"  # 작업 폴더

IMG_EXTS = (".jpg", ".jpeg", ".png")

# target 매핑: (앞번호, 뒷번호 전체) -> 한글
mapping = {}

for tf in os.listdir(target_dir):
    if not tf.lower().endswith(IMG_EXTS):
        continue
    name, _ = os.path.splitext(tf)
    # 한글 1글자 기준 분리 (뒷번호에 _숫자 포함 가능)
    m = re.match(r"^(\d+)([가-힣])(\d+(?:-\d+)?)$", name)
    if m:
        front_num = m.group(1)   # 예: 01
        hangul_ch = m.group(2)   # 예: 가
        back_num = m.group(3)    # 예: 0107 또는 0107_2
        mapping[(front_num, back_num)] = hangul_ch

# mapping의 앞 5개만 출력
# for k, v in itertools.islice(mapping.items(), 5):
for k, v in mapping.items():
    if k == ('04','0865-2'):
        print(k, "→", v)

# source 처리
for sf in os.listdir(source_dir):
    if not sf.lower().endswith(IMG_EXTS):
        continue
    name, ext = os.path.splitext(sf)
    if "_" not in name:
        continue
    left_part = name.split("_")[0]  # 예: 01-0107_2
    if "-" not in left_part:
        continue
    s_front, s_back = left_part.split("-", 1)

    key = (s_front, s_back)
    if key in mapping:
        hangul_ch = mapping[key]
        print(f"⚠ hangul_ch: {hangul_ch} key:{key}")
        new_name = name.replace("-", hangul_ch, 1) + ext
        old_path = os.path.join(source_dir, sf)
        new_path = os.path.join(source_dir, new_name)
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"변경: {sf} → {new_name} key:{key}")
        else:
            print(f"⚠ 이름 충돌: {new_name} key:{key}")
    else:
        print(f"❌ 매칭 없음: {sf} key:{key}")

print("✅ 처리 완료2")
