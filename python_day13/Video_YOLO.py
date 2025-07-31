## 비디오 프레임을 YOLO에게 전달하여 오브젝트 탐지하기
# !pip install opencv-python moviepy

## 좌에서 우로 흘러가는 다수개의 이미지를 비디오로 만들기

import cv2
import numpy as np
from glob import glob

# 출력 비디오 설정
output_width = 640
output_height = 480
fps = 30
duration_sec = 20  # ← 속도 절반으로 느리게
total_frames = fps * duration_sec

# 이미지 불러오기 및 고정 크기 리사이즈
image_paths = sorted(glob("dataset/bottle.yolov11/valid/images/*.jpg"))
resized_imgs = []

for path in image_paths[:10]:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    
    img = cv2.resize(img, (output_width, output_height))
    resized_imgs.append(img)

# 긴 배너 이미지 생성 (수평 연결)
long_img = np.hstack(resized_imgs)  # ex: (480, 6400, 3)
long_width = long_img.shape[1]

# 이동 가능 확인
if long_width <= output_width:
    raise ValueError("이미지 전체 너비가 영상보다 작습니다. 이동 불가.")

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI로 저장
video = cv2.VideoWriter("scroll_video.avi", fourcc, fps, (output_width, output_height))

if not video.isOpened():
    raise RuntimeError("VideoWriter 초기화 실패")

# → 방향으로 프레임 생성 (좌에서 우로)
for i in range(total_frames):
    # ↘ 이동 방향: 처음은 왼쪽 끝(0), 마지막은 오른쪽 끝
    dx = int((long_width - output_width) * (1 - i / total_frames))  # ← 방향 반전

    frame = long_img[:, dx:dx + output_width]

    # 오른쪽 끝 부족 시 패딩
    if frame.shape[1] != output_width:
        pad = output_width - frame.shape[1]
        frame = cv2.copyMakeBorder(frame, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=0)

    frame = frame.astype(np.uint8)
    video.write(frame)

video.release()
print("✅ scroll_video.avi 생성 완료 (←→ 방향 변경, 속도 절반)")



## YOLO 설치
# !pip install ultralytics


## 전이학습된 YOLO 모델 로드
from ultralytics import YOLO
import os

# Load the YOLO model
model = YOLO(os.path.join('C:/Users/ngins/Git/python.initial/dataset/bottle.yolov11/bottle-transfer/weights/best.pt'))
print("YOLO model loaded successfully.")



## 비디오캡쳐 비디오 라이터 생성
# 비디오 캡쳐와 비디오 라이터 생성
video_path = 'scroll_video.avi'
cap = cv2.VideoCapture(video_path)

# 비디오 파일 대신 카메라 사용
#cap = cv2.VideoCapture(0)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_video_path = 'output_scroll_video_with_detections.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("Video file opened and output video writer created successfully.")



## 전이학습된 YOLO모델을 사용하여 비디오 프레임 이미지에서 오브젝트 탐지 및 결과 비디오 저장
# 비디오 프레임을 읽어서 YOLO에서 오브젝트 탐지 및 바운딩박스 설정 후 비디오 파일에 저장
total_cnt = 200
idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Draw the results on the frame using the default plot function
    annotated_frame = results[0].plot()

    # Write the frame to the output video
    out.write(annotated_frame)
    idx += 1
    if idx == total_cnt:
        break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_video_path}")
