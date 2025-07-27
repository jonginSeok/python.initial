### YOLO, CNN 연동 이미지 분류 ###

from PIL import Image
import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO

# 모델 로드
yolo_model = YOLO('/content/drive/MyDrive/Python_AI/YOLO/Carrot Classification/carrot-transfer/weights/best.pt')



# CNN 모델 로드
try:
    # Instantiate the model first
    #cnn_crop_model = CarrotCNNWithSize()
    # Load the state dictionary
    #cnn_crop_model.load_state_dict(torch.load('/content/drive/MyDrive/Python_AI/YOLO/Carrot Classification/carrot_crop_cnn.pth'))
    #cnn_crop_model.eval()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE=", DEVICE)
    cnn_crop_model = CarrotCNNWithSize().to(DEVICE)
    cnn_crop_model.load_state_dict(torch.load('/content/drive/MyDrive/Python_AI/YOLO/Carrot Classification/carrot_crop_cnn.pth', map_location=torch.device('cpu')))
    cnn_crop_model.eval()
except FileNotFoundError:
    print("Error: CNN model not found. Please make sure 'carrot_crop_cnn.pth' exists at the specified path.")
    
    cnn_crop_model = None

# CPU/GPU 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if cnn_crop_model: 
    cnn_crop_model.to(device)
    cnn_crop_model.eval() # Set CNN model to evaluation mode

# 원본 이미지 디렉토리
input_dir = '/content/drive/MyDrive/Python_AI/YOLO/Carrot Classification/CARROT원본/train/BAD'
output_dir = 'cropped/for/cnn' # 잘라낸 이미지를 저장할 디렉토리명

os.makedirs(output_dir, exist_ok=True)  # 이미 존재해도 덮어쓰기

# 클래스 이름 맵 (YOLO output class ID to class name)
class_map = {0: 'carrot-bad', 1: 'carrot-good'}

# Image transformation for CNN input
# Assuming your CNN model expects a specific input size and normalization
# You might need to adjust the transformations based on your CNN model's requirements
transform = transforms.Compose([
    transforms.ToPILImage(), # Convert OpenCV image (numpy array) to PIL Image
    Letterbox(224),          # Resize to a standard input size for the CNN
    #transforms.Resize((224, 224)), # Letterbox에서 대신함
    transforms.ToTensor(),   # Convert PIL Image to PyTorch Tensor (scales to 0-1)
    transforms.Normalize([0.5], [0.5]) # Use the same normalization as training -1~1
])

# 이미지 순회
labels = []  # 분류된 라벨을 저장할 리스트, 정확도 검사를 위해서 사용됨

print(f"Processing images from {input_dir}...")

for img_name in os.listdir(input_dir):   # 이미지 한개씩 로드
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(input_dir, img_name)

    # YOLO 추론
    results = yolo_model(img_path)  # 리스트 리턴
    result = results[0] # 

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        continue

    # 바운딩박스 처리
    for i, box in enumerate(result.boxes):
        # 바운딩박스 좌표
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())   # 리스트의 모든 원소에 대해 int()를 적용함

        # 좌표가 이미지 영역 내에 있도록 보정함
        y1 = max(0, y1)  # 0 이상
        x1 = max(0, x1)
        y2 = min(img.shape[0], y2)  # 크기값 이하
        x2 = min(img.shape[1], x2)

        # 슬라이싱을 이용한 이미지 crop
        crop = img[y1:y2, x1:x2]   # 행, 열 슬라이싱

        # 잘려진 이미지가 내용이 없는지 
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            print(f"Warning: Empty crop for image {img_name}, box {i}. Skipping.")
            continue

        # CNN에 전달할 이미지 변환
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cropped_tensor = transform(crop_rgb).unsqueeze(0) # 배치 차원수 추가

        # 원래의 바운딩박스 크기 구하기
        crop_width = x2 - x1
        crop_height = y2 - y1
        aspect_ratio = crop_width / crop_height if crop_height > 0 else 0 # Avoid division by zero
        area = crop_width * crop_height

        # Tensor로 변환
        size_feats = torch.Tensor([crop_width, crop_height, area, aspect_ratio]).unsqueeze(0) # Add batch dimension
        size_feats = size_feats.to(device)
        cropped_tensor = cropped_tensor.to(device)

        if cnn_crop_model: # Check if cnn_model was loaded successfully
            with torch.no_grad(): # No need to calculate gradients during inference

                try:
                    output = cnn_crop_model(cropped_tensor, size_feats)

                    _, predicted_class_id = torch.max(output, dim=1)   # 값, 인덱스 리턴
                    predicted_class_name = class_map.get(predicted_class_id.item(), 'UNKNOWN')

                    #print(f"Image: {img_name}, Box {i}: Predicted class is {predicted_class_name}")
                    labels.append(predicted_class_id.item())

                except Exception as e:
                    print(f"Error during CNN inference for image {img_name}, box {i}: {e}")



### 데이터 증강(밝기 변경 후 저장) ###

import os
from PIL import Image, ImageEnhance

# 🔧 설정: 경로와 밝기 변화량
input_dir = 'C:/Users/602-00/bottle/test/broken_large'   # 실제 경로로 수정하세요
brightness_factor = 1.05             # 1.10 = +10%, 0.90 = -10%

# 변화 정도 표시 문자열 생성
brightness_label = f"br{int((brightness_factor - 1) * 100):+d}"

# 디렉토리 체크
if not os.path.isdir(input_dir):
    print(f"❌ 디렉토리 존재하지 않음: {input_dir}")
    exit()

file_count = 0

# 이미지 처리 루프
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_dir, filename)

        try:
            with Image.open(input_path) as img:
                # 밝기 조절
                enhancer = ImageEnhance.Brightness(img)
                bright_img = enhancer.enhance(brightness_factor)

                # 새 파일명 생성
                name_without_ext = os.path.splitext(filename)[0]
                output_filename = f"{name_without_ext}_{brightness_label}.jpg"
                output_path = os.path.join(input_dir, output_filename)

                # 저장
                bright_img.save(output_path)
                print(f"✅ 저장 완료: {output_filename}")
                file_count += 1

        except Exception as e:
            print(f"⚠️ 오류 발생: {filename} → {e}")

if file_count == 0:
    print("📂 처리된 이미지가 없습니다. JPG 파일이 있는지 확인하세요.")
else:
    print(f"🎉 총 {file_count}개의 파일이 밝기 조정되어 저장되었습니다.")




### 데이터 증강(이미지 회전 후 저장) ###
# 이미지 회전
import os
from PIL import Image

# 실제 이미지가 있는 경로로 수정하세요
input_dir = 'C:/Users/602-00/bottle/test/contamination'  # 예: 'C:/Users/user/Desktop/images'

# 디렉토리 존재 여부 확인
if not os.path.isdir(input_dir):
    print(f"❌ 디렉토리가 존재하지 않습니다: {input_dir}")
    exit()

file_count = 0

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # 대소문자 및 확장자 보완
        input_path = os.path.join(input_dir, filename)

        try:
            with Image.open(input_path) as img:
                # 시계 방향으로 270도 회전 (-270 또는 +90)
                rotated_img = img.rotate(90, expand=True)

                # 새 파일명 생성
                name_without_ext = os.path.splitext(filename)[0]
                output_filename = f"{name_without_ext}_rt90.jpg"
                output_path = os.path.join(input_dir, output_filename)

                # 저장
                rotated_img.save(output_path)
                print(f"✅ 저장 완료: {output_filename}")
                file_count += 1
        except Exception as e:
            print(f"⚠️ 오류 발생: {filename} → {e}")

if file_count == 0:
    print("📂 처리된 이미지가 없습니다. JPG, PNG 파일이 있는지 확인하세요.")
else:
    print(f"🎉 총 {file_count}개의 파일이 저장되었습니다.")
