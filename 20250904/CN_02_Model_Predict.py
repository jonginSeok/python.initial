from ultralytics import YOLO

# 모델 로드 (학습된 segmentation 모델 경로)
model = YOLO("runs/segment/seg_transfer3/weights/best.pt")

# 추론할 이미지 경로
source = "CarNumber.v2i.yolov8-obb/test/images/"  # 폴더 또는 단일 이미지 가능

# 추론 실행
results = model.predict(source=source, save=True, imgsz=640)

# 결과 확인
for result in results:
    print(f"Image: {result.path}")
    print(f"Classes Detected: {result.names}")
    print(f"Boxes: {result.boxes}")
    print(f"Masks: {result.masks.shape if result.masks else 'No masks'}")
