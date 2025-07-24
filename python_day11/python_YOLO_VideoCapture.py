from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow

# 1. YOLOv8 Pose 모델 로드
model = YOLO('yolo11n-pose.pt')  # yolov8s-pose.pt 또는 yolov8m-pose.pt 가능

# 2. 비디오 파일 열기
video_path = '/content/drive/MyDrive/Python_AI/YOLO/test_mp4/walk_man_left_with_phone.mp4'
cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

# 3. 출력 비디오 저장 설정 (선택사항)
output_path = 'output_pose.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
# 4. 프레임별 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count > 5:  # 100프레임까지만 처리
        break

    # YOLOv8 Pose 추론
    results = model(frame)

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 화면에 출력 (선택사항)
    #cv2.imshow('YOLO-Pose Detection', annotated_frame)
    cv2_imshow(annotated_frame)

    # 비디오 파일로 저장
    out.write(annotated_frame)

    frame_count += 1
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break

# 5. 정리
cap.release()
out.release()
#cv2.destroyAllWindows()
