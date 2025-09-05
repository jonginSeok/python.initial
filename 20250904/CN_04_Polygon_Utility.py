# 핵심은 폴리곤 → 마스크 생성 → 마스크 적용 → 크롭 → 저장 흐름입니다.

"""
🛠 처리 순서

폴리곤 좌표 준비

    예: [(x1, y1), (x2, y2), ..., (xn, yn)]
    YOLOv8‑Seg처럼 0~1 정규화 좌표라면, 이미지 크기에 맞게 변환 필요

마스크 생성

    원본 이미지 크기와 동일한 검은색(0) 마스크 생성
    cv2.fillPoly()로 폴리곤 영역을 흰색(255)으로 채움

마스크 적용

    cv2.bitwise_and()로 원본 이미지와 마스크를 합성 → 배경 제거

크롭

    cv2.boundingRect()로 폴리곤의 최소 직사각형 영역 계산
    해당 영역만 잘라내기

저장

    cv2.imwrite() 또는 PIL.Image.save()로 결과 저장
"""


import os
import cv2
import numpy as np

# ===== 경로 설정 =====
images_dir = "CarNumber.v6i.yolov8-obb/valid/images"
labels_dir = "CarNumber.v6i.yolov8-obb/valid/labels"
output_dir = "runs/crop/output"

os.makedirs(output_dir, exist_ok=True)

def yolo_to_pixel_coords(coords, img_w, img_h):
    """YOLOv8-Seg 정규화 좌표 → 픽셀 좌표 변환"""
    pixel_coords = []
    for i in range(0, len(coords), 2):
        x = float(coords[i]) * img_w
        y = float(coords[i+1]) * img_h
        pixel_coords.append([int(x), int(y)])
    return np.array(pixel_coords, dtype=np.int32)

print(f"📌 images_dir:{images_dir}")
print(f"📌 labels_dir:{labels_dir}")
print(f"📌 output_dir:{output_dir}")
# ===== 처리 루프 =====
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_dir, label_file)
    image_name = os.path.splitext(label_file)[0] + ".jpg"  # 확장자 맞게 수정 가능
    image_path = os.path.join(images_dir, image_name)

    if not os.path.exists(image_path):
        print(f"이미지 없음: {image_path}")
        continue

    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    obj_idx = 0
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        coords = parts[1:]  # x1 y1 x2 y2 ...

        polygon = yolo_to_pixel_coords(coords, img_w, img_h)

        # 마스크 생성
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # 마스크 적용
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        # 크롭
        x, y, w, h = cv2.boundingRect(polygon)

        # 유효성 체크
        if w <= 0 or h <= 0:
            print(f"[SKIP] 잘못된 크롭 영역: {image_name}, polygon={polygon.tolist()}")
            continue

        cropped_img = masked_img[y:y+h, x:x+w]

        if cropped_img is None or cropped_img.size == 0:
            print(f"[SKIP] 크롭 결과가 비어있음: {image_name}")
            continue

        # 저장
        base_name = os.path.splitext(image_name)[0]
        
        os.makedirs(output_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(output_dir, f"{base_name}_obj{obj_idx}_mask.jpg"), masked_img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_obj{obj_idx}_crop.jpg"), cropped_img)

        obj_idx += 1

print("✅ 처리 완료!")

