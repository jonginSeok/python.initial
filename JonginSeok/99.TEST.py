import os
import json

def yolo_to_coco(yolo_dir, image_dir):
    annotations = []
    image_id = 0
    ann_id = 0

    for filename in os.listdir(yolo_dir):
        if not filename.endswith('.txt'):
            continue
        image_id += 1
        txt_path = os.path.join(yolo_dir, filename)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            cls, x, y, w, h = map(float, line.strip().split())
            # 정규화된 좌표를 COCO bbox로 변환
            x_min = x - w / 2
            y_min = y - h / 2
            bbox = [x_min, y_min, w, h]

            annotations.append({
                "image_id": filename.replace('.txt', ''),
                "category_id": int(cls),
                "bbox": bbox,
                "id": ann_id
            })
            ann_id += 1

    with open('ground_truth.json', 'w') as f:
        json.dump(annotations, f, indent=2)

yolo_to_coco('JonginSeok/dataset/labels', 'JonginSeok/dataset/images')