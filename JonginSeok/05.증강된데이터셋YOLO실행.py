import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    from ultralytics import YOLO

    print("True여야 GPU 사용 가능 :", torch.cuda.is_available())
    print(f"사용 가능한 GPU({device}) 수:", torch.cuda.device_count())

    # 기존 모델 불러오기 (COCO 학습됨)
    model = YOLO("yolo11n.pt")

    model.train(
        data="JonginSeok/dataset/data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        project="JonginSeok/dataset/result",
        name="yolo11n_bottle_4class",
        verbose=True,  # output
    )

    # print(f'# result : {result}')

    val_result = model.val()
    metrics = val_result.metrics

    print("📌 클래스별 mAP:")
    for i, class_name in enumerate(val_result.names.values()):
        print(
            f"{class_name}: mAP50 = {metrics['map50_per_class'][i]:.3f}, mAP50-95 = {metrics['map_per_class'][i]:.3f}"
        )
