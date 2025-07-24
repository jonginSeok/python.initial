# ====== 전이학습 후 학습된 모델을 사용하여 병 이미지 분류 테스트 ======

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# 1. 경로 설정
model_path = 'C:/Users/ngins/Git/python.initial/runs/detect/yolo11n_add_cup_class16/weights/best.pt'
image_path = 'C:/Users/ngins/Downloads/bottle.tar/bottle/test/broken_large/000.png'

# 2. 모델 로드
model = YOLO(model_path)

# 3. 이미지 추론
results = model(image_path)

# 4. 바운딩 박스가 그려진 결과 이미지 얻기
# 결과는 list 형태이므로 첫 번째 결과만 사용
print(results)  # 결과 확인
res_plotted = results[0].plot()  # numpy.ndarray (BGR)

# 5. 시각화 (OpenCV는 BGR, matplotlib은 RGB)
res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(res_rgb)
plt.axis('off')
plt.title('Detection Result')
plt.show()
