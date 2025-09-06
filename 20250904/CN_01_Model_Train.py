from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s-seg.pt')  # 사전 학습된 세그멘테이션 모델
    model.train(
        data='20250904/custom_seg.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='seg_transfer',
        device='cpu'  # GPU 사용
    )











"""
5️⃣ 학습 결과 확인
학습이 완료되면 runs/segment/seg_transfer/weights/best.pt에 최적 모델이 저장됩니다.
"""
results = model.val()
print(results)

"""
[결과]
100 epochs completed in 0.296 hours.
Optimizer stripped from runs\segment\seg_transfer3\weights\last.pt, 23.9MB
Optimizer stripped from runs\segment\seg_transfer3\weights\best.pt, 23.9MB

Validating runs\segment\seg_transfer3\weights\best.pt...
Ultralytics 8.3.173  Python-3.13.5 torch-2.8.0+cu129 CUDA:0 (NVIDIA GeForce RTX 2070 Super with Max-Q Design, 8192MiB)
YOLOv8s-seg summary (fused): 85 layers, 11,780,374 parameters, 0 gradients, 42.4 GFLOPs
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100%|██████████| 3/3 [00:01<00:00,  1.89it/s]
                  all         86        176      0.999          1      0.995      0.904      0.993      0.994      0.994      0.741
        license_plate         86         86      0.999          1      0.995      0.895      0.987      0.988      0.992      0.562
                 text         86         90      0.999          1      0.995      0.913      0.999          1      0.995       0.92
Speed: 0.3ms preprocess, 3.3ms inference, 0.0ms loss, 4.7ms postprocess per image
Results saved to runs\segment\seg_transfer3
"""
