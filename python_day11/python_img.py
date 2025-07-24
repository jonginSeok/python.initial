import cv2
import os
import glob

# 🗂️ 설정: 원본 이미지 폴더 및 저장 위치
input_folder = 'C:\\Users\\ngins\\Downloads\\Bottle Detection.v4i.yolov11\\train\\images\\'        # 원본 이미지 폴더 경로
output_folder = 'C:\\Users\\ngins\\Downloads\\Bottle Detection.v4i.yolov11\\train\\images\\ratio50\\' #'your_output_folder_path'  # 저장할 폴더 경로

input_labels_folder = 'C:\\Users\\ngins\\Downloads\\Bottle Detection.v4i.yolov11\\train\\labels\\'
output_labels_folder = 'C:\\Users\\ngins\\Downloads\\Bottle Detection.v4i.yolov11\\train\\labels\\ratio50\\'

# 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# 이미지 확장자 필터링
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_folder, ext)))

print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")

# 라벨 파일도 같은 방식으로 가져오기
label_extensions = ['*.txt']
label_files = []
for ext in label_extensions:
    label_files.extend(glob.glob(os.path.join(input_labels_folder, ext)))

# 🌀 회전 함수 정의
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 여유 공간을 주기 위해 회전 후 전체 크기 고려
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

for label_file in label_files:
    with open(label_file, 'r') as f:
        lines = f.readlines()

    # 라벨 파일에서 클래스와 좌표 추출
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # 잘못된 형식의 라벨은 무시
        class_id = parts[0]
        x_center, y_center, width, height = map(float, parts[1:5])
        new_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # 새 라벨 파일 저장
    label_name = os.path.basename(label_file)
#    new_label_file = os.path.join(output_folder, label_name)
    new_label_file = os.path.join(output_labels_folder, f"{label_name}_rat50.txt")
    
    with open(new_label_file, 'w') as f:
        f.writelines(new_lines)

# 🔁 이미지 하나씩 처리 및 저장
for file in image_files:
    img = cv2.imread(file)
    if img is None:
        print(f"이미지 로딩 실패: {file}")
        continue

    rotated = rotate_image(img, 50)  # 30도 회전

    # 🔤 저장 파일명 생성
    base = os.path.basename(file)
    name, ext = os.path.splitext(base)
    print(f"처리 중: {file} -> {os.path} {name}_rat30{ext}")
    output_file = os.path.join(output_folder, f"{name}_rat50{ext}")

    cv2.imwrite(output_file, rotated)
    print(f"저장 완료: {output_file}")