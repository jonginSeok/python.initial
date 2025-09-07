"""
단일 이미지와 라벨로 바운딩 박스 그리기
아래 코드는 이미지 파일과 YOLO 라벨(.txt)을 입력받아 바운딩 박스(및 세그멘테이션 폴리곤이 있으면 그것도) 를 그려 저장합니다. 
YOLO 탐지 형식(class xc yc w h)과 세그멘테이션 형식(class x1 y1 x2 y2 …) 모두 지원합니다.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

# -----------------------------
# 유틸리티
# -----------------------------


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def yolo_xywhn_to_xyxy(xc, yc, w, h, W, H):
    # 정규화 -> 픽셀 좌표 (좌상단-우하단)
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def polygon_norm_to_pixels(points_norm: List[float], W: int, H: int) -> np.ndarray:
    # [x1, y1, x2, y2, ...] 정규화 → 픽셀 좌표 Nx2
    pts = []
    for i in range(0, len(points_norm), 2):
        x = int(clamp01(points_norm[i]) * W)
        y = int(clamp01(points_norm[i+1]) * H)
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)

# -----------------------------
# 라벨 파서 (탐지/세그멘테이션 모두 지원)
# -----------------------------


def parse_yolo_label_file(label_path: str, img_w: int, img_h: int):
    """
    각 라인 파싱:
        - 탐지 형식: class xc yc w h [conf]
        - 세그 형식: class x1 y1 x2 y2 ... (짝수 개의 좌표쌍)
    반환: records = [{ 'cls': int, 'bbox': (x1,y1,x2,y2), 'poly': np.ndarray|None, 'conf': float|None }]
    """
    records = []
    if not os.path.exists(label_path):
        return records

    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            nums = list(map(float, parts[1:]))

            # 세그멘테이션: 클래스 다음 숫자 개수가 짝수(좌표쌍)
            if len(nums) >= 6 and len(nums) % 2 == 0:
                # polygon
                poly = polygon_norm_to_pixels(nums, img_w, img_h)
                x1, y1 = np.min(poly, axis=0)
                x2, y2 = np.max(poly, axis=0)
                records.append({
                    "cls": cls_id,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "poly": poly,
                    "conf": None
                })
            else:
                # 탐지 (xc,yc,w,h) [+ conf]
                xc, yc, w, h = nums[:4]
                conf = nums[4] if len(nums) >= 5 else None
                x1, y1, x2, y2 = yolo_xywhn_to_xyxy(xc, yc, w, h, img_w, img_h)
                records.append({
                    "cls": cls_id,
                    "bbox": (x1, y1, x2, y2),
                    "poly": None,
                    "conf": conf
                })
    return records

# -----------------------------
# 그리기
# -----------------------------


def draw_boxes_on_image(
    image_path: str,
    label_path: str,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    draw_polygon: bool = True
):
    """
    이미지와 YOLO 라벨을 읽어 바운딩박스/폴리곤 및 클래스명(확신도)까지 오버레이 후 저장/표시
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {image_path}")

    H, W = img.shape[:2]
    recs = parse_yolo_label_file(label_path, W, H)

    # 색상 팔레트 (cls_id별 고정 색)
    palette = [
        (0, 204, 255),   # 하늘
        (0, 255, 0),     # 초록
        (255, 178, 0),   # 주황
        (255, 51, 51),   # 빨강
        (153, 102, 255),  # 보라
        (0, 136, 170),   # 청록
    ]

    # 두께/폰트 스케일 자동화
    thickness = max(2, int(0.002 * (W + H)))
    font_scale = max(0.4, 0.0008 * (W + H))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for r in recs:
        cls_id = r["cls"]
        color = palette[cls_id % len(palette)]
        x1, y1, x2, y2 = r["bbox"]

        # 박스
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 폴리곤
        if draw_polygon and r["poly"] is not None and len(r["poly"]) >= 3:
            cv2.polylines(img, [r["poly"]], isClosed=True,
                          color=color, thickness=thickness)

        # 라벨 텍스트
        cls_name = class_names[cls_id] if (
            class_names and 0 <= cls_id < len(class_names)) else str(cls_id)
        if r["conf"] is not None:
            label = f"{cls_name} {r['conf']:.2f}"
        else:
            label = f"{cls_name}"

        # 텍스트 배경 박스
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        bx1, by1 = x1, max(0, y1 - th - 6)
        bx2, by2 = x1 + tw + 6, y1
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # 저장 또는 반환
    if save_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(os.path.dirname(
            image_path), f"{base}_drawn.jpg")

    ok = cv2.imwrite(save_path, img)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {save_path}")
    return save_path

# -----------------------------
# 예시 사용
# -----------------------------
# if __name__ == "__main__":
#     image_path = r"C:/Users/ngins/Git/python.initial/20250904/CarNumber.v4i.yolov8-obb/train/images/01-0216_jpg.rf.f7d2af51f8befe087217974e86d56f3a.jpg"
#     label_path = r"C:/Users/ngins/Git/python.initial/20250904/CarNumber.v4i.yolov8-obb/train/labels/01-0216_jpg.rf.f7d2af51f8befe087217974e86d56f3a.txt"
#     class_names = ["license_plate", "text"]  # 0, 1 클래스명

#     out = draw_boxes_on_image(
#         image_path=image_path,
#         label_path=label_path,
#         save_path=r"C:/Users/ngins/Git/python.initial/20250904/overlay/01-0216_jpg.rf.f7d2af51f8befe087217974e86d56f3a_drawn.jpg",
#         class_names=class_names,
#         draw_polygon=True
#     )
#     print("저장:", out)


"""
디렉터리 일괄 처리 옵션
여러 장을 한 번에 처리하고 싶다면 아래 헬퍼를 추가해 사용하세요.
"""
if __name__ == "__main__":
    def draw_folder(
        images_dir: str,
        labels_dir: str,
        out_dir: str,
        class_names: Optional[List[str]] = None,
        draw_polygon: bool = True,
        image_exts=(".jpg", ".jpeg", ".png")
    ):
        os.makedirs(out_dir, exist_ok=True)
        names = [n for n in os.listdir(
            images_dir) if n.lower().endswith(image_exts)]
        names.sort()
        for n in names:
            img_path = os.path.join(images_dir, n)
            txt_path = os.path.join(
                labels_dir, os.path.splitext(n)[0] + ".txt")
            if not os.path.exists(txt_path):
                print(f"[SKIP] 라벨 없음: {n}")
                continue
            save_path = os.path.join(
                out_dir, os.path.splitext(n)[0] + "_drawn.jpg")
            try:
                draw_boxes_on_image(img_path, txt_path,
                                    save_path, class_names, draw_polygon)
                print(f"[OK] {n}")
            except Exception as e:
                print(f"[ERR] {n} → {e}")

# 사용 예시
images_path = r"20250904/CarNumber.v4i.yolov8-obb/train/images"
labels_path = r"20250904/CarNumber.v4i.yolov8-obb/train/labels"
overlay_path = r"20250904/runs/mask_draw"
draw_folder(images_path, labels_path, overlay_path, ["license_plate", "text"])

"""
체크 포인트
    - 라벨 형식 자동 감지: 줄당 숫자 개수로 탐지/세그멘테이션을 구분합니다.
    - 정규화 좌표 처리: 0–1 범위를 이미지 크기에 맞게 픽셀 좌표로 변환합니다.
    - 클래스명/확신도: 클래스명이 있으면 표기, conf 값이 있으면 같이 표기합니다.
    - 시각 요소 자동 스케일: 이미지 크기에 비례해 두께/폰트 크기를 조절합니다.
"""
