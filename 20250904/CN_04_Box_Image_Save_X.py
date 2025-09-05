"""
목적과 전제
4번 “바운딩 박스를 오려내서 이미지 파일에 저장”을 자동화하는 코드를 제공합니다. 한 장의 원본 이미지와 해당 YOLO 라벨(.txt)을 받아 다음을 저장합니다:
    - plate: 번호판 전체 영역 (class 0)
    - s1: 단행 번호판의 텍스트 라인 (class 1이 1개일 때)
    - d1, d2: 2행 번호판의 텍스트 라인들 (class 1이 2개일 때, 위쪽이 d1, 아래쪽이 d2)

전제:
    - YOLO 라벨 형식: class x_center y_center width height (모두 0~1 정규화). 세그멘테이션 라벨이어도 첫 5개 값은 bbox여야 함.
    - 클래스 정의: 0 = license_plate(1개), 1 = text(1~2개, 라인 단위).
    - 파일명 규칙: origin_name_{plate|s1|d1|d2}.jpg (예: 서울32가1234_plate.jpg, 서울32가1234_d1.jpg, 서울32가1234_d2.jpg)

"""


import os
import cv2
from typing import List, Tuple, Optional

# -----------------------------
# 좌표/크롭 유틸
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def yolo_norm_to_xyxy(xc, yc, w, h, W, H):
    # YOLO 정규화 좌표 -> 픽셀 절대좌표 (x1, y1, x2, y2)
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    x2 = clamp(x2, 0, W - 1)
    y2 = clamp(y2, 0, H - 1)
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

def crop(img, box, margin=0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    if margin > 0:
        xm = int((x2 - x1) * margin)
        ym = int((y2 - y1) * margin)
        x1 = clamp(x1 - xm, 0, w - 1)
        y1 = clamp(y1 - ym, 0, h - 1)
        x2 = clamp(x2 + xm, 0, w - 1)
        y2 = clamp(y2 + ym, 0, h - 1)
    return img[y1:y2, x1:x2]

# -----------------------------
# 라벨 파서
# -----------------------------
def parse_yolo_label(label_path: str) -> Tuple[Optional[Tuple[float,float,float,float]], List[Tuple[float,float,float,float]]]:
    """
    returns:
      plate_box (xc, yc, w, h) or None
      text_boxes: list of (xc, yc, w, h)
    """
    plate_box = None
    text_boxes = []
    if not os.path.exists(label_path):
        return None, []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            if cls_id == 0:
                # plate는 1개만 사용
                plate_box = (xc, yc, w, h)
            elif cls_id == 1:
                text_boxes.append((xc, yc, w, h))
    return plate_box, text_boxes

# -----------------------------
# 단일 이미지 처리
# -----------------------------
def process_one_image(
    image_path: str,
    label_path: str,
    out_dir: str,
    margin_ratio_plate: float = 0.03,
    margin_ratio_text: float = 0.05
):
    """
    image_path / label_path를 읽어 plate, s1 또는 d1/d2를 저장
    out_dir 하위에 파일 저장
    """
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[SKIP] 이미지 로드 실패: {image_path}")
        return None

    H, W = img.shape[:2]
    base = os.path.splitext(os.path.basename(image_path))[0]

    plate_norm, text_norms = parse_yolo_label(label_path)

    # 유효성 검사
    if plate_norm is None:
        print(f"[WARN] plate(class 0) 없음 → 건너뜀: {image_path}")
        return None
    if not (1 <= len(text_norms) <= 2):
        print(f"[WARN] text(class 1) 개수 {len(text_norms)} (허용: 1~2) → 건너뜀: {image_path}")
        return None

    # plate 크롭
    px1, py1, px2, py2 = yolo_norm_to_xyxy(*plate_norm, W, H)
    plate_crop = crop(img, (px1, py1, px2, py2), margin=margin_ratio_plate)
    plate_out = os.path.join(out_dir, f"{base}_plate.jpg")
    cv2.imwrite(plate_out, plate_crop)

    # text 크롭: d/s 규칙 적용
    # - 1개면 s1
    # - 2개면 y 기준 위/아래 정렬해 d1(위), d2(아래)
    text_boxes_xy = []
    for (xc, yc, w, h) in text_norms:
        x1, y1, x2, y2 = yolo_norm_to_xyxy(xc, yc, w, h, W, H)
        text_boxes_xy.append((x1, y1, x2, y2))

    # plate 내부에서만 자르고 싶다면, 교차 영역으로 클리핑
    def clip_to_plate(box, plate):
        x1, y1, x2, y2 = box
        px1, py1, px2, py2 = plate
        x1 = clamp(x1, px1, px2)
        x2 = clamp(x2, px1, px2)
        y1 = clamp(y1, py1, py2)
        y2 = clamp(y2, py1, py2)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    # 정렬: 박스 top(y1) 기준 오름차순
    text_boxes_xy.sort(key=lambda b: b[1])

    saved = {"plate": plate_out}

    if len(text_boxes_xy) == 1:
        b = clip_to_plate(text_boxes_xy[0], (px1, py1, px2, py2))
        if b is None:
            print(f"[WARN] text가 plate 영역과 교차하지 않음 → s1 저장 생략: {image_path}")
        else:
            s1_img = crop(img, b, margin=margin_ratio_text)
            s1_out = os.path.join(out_dir, f"{base}_s1.jpg")
            cv2.imwrite(s1_out, s1_img)
            saved["s1"] = s1_out
    elif len(text_boxes_xy) == 2:
        b1 = clip_to_plate(text_boxes_xy[0], (px1, py1, px2, py2))
        b2 = clip_to_plate(text_boxes_xy[1], (px1, py1, px2, py2))
        # 교차 실패 방어
        pairs = []
        if b1 is not None: pairs.append(("d1", b1))
        if b2 is not None: pairs.append(("d2", b2))
        for tag, bb in pairs:
            crop_img = crop(img, bb, margin=margin_ratio_text)
            out_path = os.path.join(out_dir, f"{base}_{tag}.jpg")
            cv2.imwrite(out_path, crop_img)
            saved[tag] = out_path

    print(f"[OK] 저장: {saved}")
    return saved

# -----------------------------
# 폴더 단위 일괄 처리
# -----------------------------
def batch_process(
    images_dir: str,
    labels_dir: str,
    out_dir: str,
    image_exts=(".jpg", ".jpeg", ".png")
):
    os.makedirs(out_dir, exist_ok=True)
    all_imgs = [f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)]
    all_imgs.sort()

    stats = {"ok": 0, "skip": 0}
    for img_name in all_imgs:
        image_path = os.path.join(images_dir, img_name)
        base = os.path.splitext(img_name)[0]
        label_path = os.path.join(labels_dir, f"{base}.txt")
        res = process_one_image(image_path, label_path, out_dir)
        if res is None:
            stats["skip"] += 1
        else:
            stats["ok"] += 1

    print(f"\n완료: OK={stats['ok']} / SKIP={stats['skip']}")


"""
실행 예시
"""

# 예시 1) 학습 데이터(ground-truth)에서 크롭
# images_dir = r"c:/data/plates/images/val"
# labels_dir = r"c:/data/plates/labels/val"
# out_dir    = r"c:/test/text_images"
# batch_process(images_dir, labels_dir, out_dir)

# 예시 2) YOLO 추론 결과에서 크롭
# Ultralytics predict의 기본 구조:
# runs/segment/predict/                      ← 이미지가 저장되는 폴더
# runs/segment/predict/labels/               ← 라벨 텍스트(정규화 bbox) 폴더
# images_dir = r"runs/segment/predict"
# labels_dir = r"runs/segment/predict/labels"
# out_dir    = r"c:/test/text_images"
# batch_process(images_dir, labels_dir, out_dir)


#예시 1) 실행
images_dir = r"C:/Users/ngins/Git/python.initial/20250904/CarNumber.v4i.yolov8-obb/train/images"
labels_dir = r"C:/Users/ngins/Git/python.initial/20250904/CarNumber.v4i.yolov8-obb/train/labels"
out_dir    = r"C:/Users/ngins/Git/python.initial/20250904/runs/crop_images"
batch_process(images_dir, labels_dir, out_dir)


"""
주의사항과 팁
    클래스 개수 검증:
        - class 0: 반드시 1개여야 저장 진행.
        - class 1: 1개면 s1만 저장, 2개면 d1/d2 저장. 0개 또는 3개 이상이면 스킵.

    라인 정렬 기준:
        - 위쪽 라인 → d1, 아래쪽 라인 → d2. y1(상단) 좌표로 정렬합니다.

    plate 내부 클리핑:
        - 텍스트 박스가 plate 밖으로 일부 나간 경우 plate 영역으로 클리핑 후 저장해 가장 깔끔하게 만듭니다.

    마진 조절:
        - 잘림을 방지하려면 margin_ratio_plate/text를 0.03~0.1 사이에서 데이터에 맞춰 조절하세요.

    파일명 규칙:
        - 원본 파일명이 “서울32가1234.jpg”면, 저장 결과는

            - 서울32가1234_plate.jpg
            - 서울32가1234_s1.jpg 또는 서울32가1234_d1.jpg, 서울32가1234_d2.jpg
"""