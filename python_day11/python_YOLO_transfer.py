
if __name__ == '__main__':
    from ultralytics import YOLO

    # 기존 모델 불러오기 (COCO 학습됨)
    # model = YOLO('yolo11n.pt')
    model = YOLO('yolo11n.pt') # 각자의 경로

    # 새 클래스 1개만 포함한 데이터로 추가 학습
    # model.train(
    #     data='data.yaml',
    #     epochs=50,
    #     imgsz=640,
    #     batch=16,
    #     name='yolo11n_add_cup_class'
    # )

    # 2025.07.24 add
    model.train(
        data='C:\\Users\\ngins\\Downloads\\Bottle Detection.v4i.yolov11\\data.yaml',
        epochs=10,
        imgsz=640,
        batch=16,     # 메모리 문제로 배치 사이즈 줄임
        project = '',
        name='yolo11n_add_cup_class',
        pretrained=True,
        # patience=10, # 정확도(es_metric)가 10번을 넘기면 그만
        # es_metric='metrics/mAP50-95(B)'   # mAP50' # old version
        # verbose=True,  # 학습 과정 출력
    )
