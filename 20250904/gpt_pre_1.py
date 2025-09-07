"""
🚗 전체 워크플로우 단계별 가이드
1️⃣ YOLOv8-Seg 모델 전이학습 및 mAP 확인
-데이터 준비: Roboflow에서 export한 segmentation 형식의 데이터셋을 YOLOv8 형식으로 다운로드
-모델 설정: Ultralytics 라이브러리 설치 후 YOLOv8-Seg 모델 로드 (yolov8s-seg.pt 등)
-전이학습 실행:

[Bash]
yolo task=segment mode=train model=yolov8s-seg.pt data=data.yaml epochs=100 imgsz=640

성능 확인: 학습 후 results.csv 또는 runs/segment/train/ 내 metrics 확인 → mAP50, mAP50-95 등
"""
import os
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s-seg.pt')  # 사전 학습된 세그멘테이션 모델
    model.train(
        data='custom_seg.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='seg_transfer',
        device='cuda'  # GPU 사용
    )
# results = model.val()
# print(results)

"""
2️⃣ 테스트 이미지에 바운딩 박스 시각화
추론 실행:

[Bash]
yolo task=segment mode=predict model=best.pt source=test_images/ save=True

결과 확인: runs/segment/predict/ 폴더에 바운딩 박스 및 마스크가 적용된 이미지 저장됨
번호판 내부 텍스트만 인식되는지 확인: 시각적으로 확인하거나, 클래스별로 필터링해서 분석

"""

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

"""
3️⃣ 바운딩 박스 오려내기 및 이미지 저장
OpenCV 사용 예시:
"""


def crop_and_save(image_path, bbox, save_path):
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = bbox  # 바운딩 박스 좌표
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(save_path, cropped)


"""
파일명 규칙:
- 번호판 전체: origin_name_plate.jpg
- 단행 텍스트: origin_name_s1.jpg
- 2행 텍스트: origin_name_d1.jpg, origin_name_d2.jpg




















4️⃣ 이미지 라벨 생성
CTC 모델용 라벨 형식: 일반적으로 .txt 파일에 이미지와 동일한 이름으로 저장

[txt]
origin_name_plate.jpg: 12가3456
origin_name_s1.jpg: 서울
origin_name_d1.jpg: 12가
origin_name_d2.jpg: 3456

- 주의사항: CTC 모델은 시퀀스 기반이므로, 라벨은 정확한 순서와 텍스트로 구성되어야 함


5️⃣ CTC 모델 학습 및 mAP 확인
모델 예시: CRNN, TrOCR, 또는 커스텀 CTC 기반 모델

학습 코드 예시 (PyTorch 기반):

[python]
# 이미지와 라벨 로딩 → CTC Loss 적용
# 모델 학습 후 validation set에서 accuracy 또는 CER/WER 계산

성능 평가: mAP 외에도 Character Error Rate (CER) 또는 Word Error Rate (WER) 사용 가능


6️⃣ 학습된 CTC 모델로 번호판 텍스트 인식
추론 실행:

"""
model.eval()
with torch.no_grad():
    output = model(image_tensor)
    decoded_text = decode_ctc_output(output)
    print("인식된 번호판:", decoded_text)

"""
결과 저장: 이미지와 함께 인식된 텍스트를 .csv 또는 .json으로 저장 가능
"""
