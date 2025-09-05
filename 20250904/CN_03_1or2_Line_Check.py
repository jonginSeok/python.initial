import os
import csv

def check_plate_rows_from_yolo_label(label_path):
    """
    YOLO 형식 라벨 파일을 읽어 1행/2행 번호판 판별
    label_path: YOLO txt 라벨 파일 경로
    return: (행 수, 경고 메시지)
    """
    class_counts = {0: 0, 1: 0}

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id in class_counts:
                class_counts[cls_id] += 1 # 한개의 파일당 line별로

    # 조건 체크
    if class_counts[0] != 1:
        return None, f"class 0 개수 {class_counts[0]}개 (1개여야 함)"
    if not (1 <= class_counts[1] <= 2):
        return None, f"class 1 개수 {class_counts[1]}개 (1~2개여야 함)"

    # 행 수 판별
    if class_counts[1] == 1:
        return 1, None
    elif class_counts[1] == 2:
        return 2, None

def process_labels_folder(labels_folder, output_csv):
    results = []
    for file_name in os.listdir(labels_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(labels_folder, file_name)
            rows, warning = check_plate_rows_from_yolo_label(file_path)
            results.append({
                "file": file_name,
                "rows": rows,
                "warning": warning
            })

    # CSV 저장
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["file", "rows", "warning"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"처리 완료! 결과가 '{output_csv}'에 저장되었습니다.")

# 사용 예시
gubun_val = "test"
labels_dir = "CarNumber.v2i.yolov8-obb/"+gubun_val+"/labels"  # YOLO 라벨 폴더 경로
output_file = "runs/plate_rows_result/plate_rows_result_"+gubun_val+".csv"
process_labels_folder(labels_dir, output_file)
