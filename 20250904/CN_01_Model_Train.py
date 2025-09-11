from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s-seg.pt')  # 사전 학습된 세그멘테이션 모델
    model.train(
        data='20250904/custom_seg.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='20250904/runs/segment/seg_transfer',
        device='cuda'  # GPU 사용
    )
