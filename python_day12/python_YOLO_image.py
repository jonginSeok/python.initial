import torch
torch.cuda.is_available()
#### 이미지에서 해당 오브젝트만 오려내서 파일에 저장하기 ####
from PIL import Image
import os
import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO

# Classification: 분류, 등급

# 모델 로드
yolo_model = YOLO('/content/drive/MyDrive/Python_AI/YOLO/yolo11n_add_cup_class9/weights/best.pt')
# 원본 이미지 디렉토리
input_dir = '/content/drive/MyDrive/Python_AI/YOLO/CarrotClassification/Bottle/train/BAD'
output_dir = '/content/drive/MyDrive/Python_AI/YOLO/CarrotClassification/Bottle/cropped/BAD'

os.makedirs(output_dir, exist_ok=True)

# 클래스 이름 → CNN용 폴더명 지정
class_map = {0: 'carrot-bad', 1: 'carrot-good'}

# 이미지 순회
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(input_dir, img_name)
    results = yolo_model(img_path)
    result = results[0]

    img = cv2.imread(img_path)

    # 박스마다 잘라서 저장
    for i, box in enumerate(result.boxes):   # 한 이미지에 다수개가 탐지될 수 있음
        cls_id = int(box.cls[0])
        class_name = class_map[cls_id]

        # 바운딩 박스 좌표
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 첫번째 바운딩박스의 좌상,우하 좌표, Tensor를 리스트로...
        crop = img[y1:y2, x1:x2]

        save_dir = os.path.join(output_dir, class_name)   #저장 디렉토리 안에 클래스이름의 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)

        # 바운딩 박스 이미지를 파일에 저장
        save_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_{i}.jpg")  # CNN은 jpg 포맷에서 가장 효율적인 성능을 낸다
        cv2.imwrite(save_path, crop)