import os
import csv
from PIL import Image


"""
0 : license_plate
1 : text 
"""

# 클래스 매핑
class_map = {
    0: "license_plate",
    1: "text"
}

# 마스크 좌표 → 바운딩 박스 변환


def polygon_to_bbox(points, img_w, img_h):
    xs = points[0::2]
    ys = points[1::2]
    abs_x = [x * img_w for x in xs]
    abs_y = [y * img_h for y in ys]
    xmin, xmax = min(abs_x), max(abs_x)
    ymin, ymax = min(abs_y), max(abs_y)
    return round(xmin), round(ymin), round(xmax), round(ymax)


# GT 변환
label_dir = "20250904/CarNumber.v6i.yolov8-obb/valid/labels"   # YOLOv8n-seg 라벨 폴더
image_dir = "20250904/CarNumber.v6i.yolov8-obb/valid/images"   # 원본 이미지 폴더

with open("20250904/runs/gt_boxes.txt", "w", newline="") as f:
    writer = csv.writer(f)
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        img_name = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue
        img_w, img_h = Image.open(img_path).size

        with open(os.path.join(label_dir, label_file), "r") as lf:
            for line in lf:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                polygon = parts[1:]
                xmin, ymin, xmax, ymax = polygon_to_bbox(polygon, img_w, img_h)
                writer.writerow(
                    [img_name, xmin, ymin, xmax, ymax, class_map[cls_id]])

print("[INFO] gt_boxes.txt 생성 완료")
