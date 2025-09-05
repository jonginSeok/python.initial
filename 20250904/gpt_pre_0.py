# Car Number Images 500 Annotation

"""
YOLOv8s-seg.pt는 이미지 세그멘테이션(Segmentation) 작업을 위한 Ultralytics YOLOv8 모델입니다. 
Python에서 실행하려면 먼저 필요한 라이브러리를 설치하고, 모델을 불러와 예측을 수행하는 코드를 작성해야 합니다. 
아래에 단계별로 정리해드릴게요 👇
"""

"""
🛠️ 1. 설치 및 Import
먼저 YOLOv8 라이브러리를 설치합니다:

[Bash]
pip install ultralytics

그 다음 Python 코드에서 필요한 모듈을 import 합니다:
"""
from ultralytics import YOLO

"""
🚀 2. 모델 로드 및 실행 예제
이미지 세그멘테이션을 위한 실행 예제입니다:
"""

# # 모델 로드
# model = YOLO('yolov8s-seg.pt')  # 세그멘테이션용 모델

# # 이미지 예측
# results = model.predict(source='your_image.jpg', conf=0.3, save=True)

# # 결과 확인
# for result in results:
#     result.show()  # 결과 이미지 보기
#     result.save_txt()  # 텍스트로 저장 (선택 사항)

"""
source: 예측할 이미지 경로
conf: confidence threshold (예: 0.3)
save=True: 결과 이미지 저장

📦 3. CLI(Command Line) 방식도 가능
터미널에서 직접 실행할 수도 있어요:

[bash]
yolo task=segment mode=predict model=yolov8s-seg.pt source=your_image.jpg conf=0.3

이 방식은 결과를 runs/segment/predict 폴더에 저장합니다.



🧠 YOLOv8 전이 학습 예제 (Segmentation 기준)
1️⃣ 환경 준비

[bash]
pip install ultralytics

[python]
from ultralytics import YOLO

2️⃣ 데이터셋 준비
데이터셋 구조는 다음과 같아야 합니다:

[코드]
datasets/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/

images/: 원본 이미지
labels/: YOLO 포맷의 라벨 텍스트 파일 (세그멘테이션은 .txt에 polygon 정보 포함)

3️⃣ YAML 파일 생성
예: custom_seg.yaml

[yaml]
train: datasets/train/images
val: datasets/val/images

nc: 2  # 클래스 수
names: ['apple', 'banana']  # 클래스 이름


4️⃣ 전이 학습 코드

[python]
model = YOLO('yolov8s-seg.pt')  # 사전 학습된 세그멘테이션 모델

model.train(
    data='custom_seg.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='seg_transfer',
    device='cuda'  # GPU 사용
)

data: YAML 파일 경로

epochs: 학습 반복 횟수

imgsz: 이미지 크기

batch: 배치 사이즈

device: 'cuda' 또는 'cpu'

5️⃣ 학습 결과 확인
학습이 완료되면 runs/segment/seg_transfer/weights/best.pt에 최적 모델이 저장됩니다.

[python]
results = model.val()
print(results)


🔍 참고 자료
벨로그 YOLOv8 예제(https://velog.io/@spacenet/YOLO-v8-easy-to-use)에서는 데이터셋 구성과 YAML 설정, 하이퍼파라미터 조정까지 상세히 설명되어 있어요.

네이버 블로그 예제(https://blog.naver.com/PostView.naver?blogId=beyondlegend&logNo=223050797442)에서는 Roboflow 데이터셋을 활용한 학습 흐름을 잘 보여줍니다.

필요하시면 Roboflow에서 커스텀 세그멘테이션 데이터셋을 직접 만들거나 가져오는 방법도 알려드릴게요. 어떤 객체를 세그멘테이션하고 싶으신가요? 🍎📦
"""
