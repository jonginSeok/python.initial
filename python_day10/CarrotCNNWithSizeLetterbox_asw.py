import torch

from CarrotCNNWithSizeLetterbox import CarrotCNNWithSize, Letterbox, transform
# CarrotCNNWithSize 클래스와 Letterbox 클래스를 가져옴
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE=", DEVICE)

label_map = {'BAD': 0, 'GOOD': 1}  # class_names = ['BAD', 'GOOD']
class_names = list(label_map.keys())  # ['BAD', 'GOOD']
data_path = "C:\\Users\\ngins\\Git\\python.initial\\dataset\\dental\\images"

model = CarrotCNNWithSize().to(DEVICE)

# 모델 저장 (학습 루프 끝난 후)
save_path = data_path + '\\Classification\\dental_cnn_with_size_letterbox.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")


# 모델 다시 불러오기
save_path = data_path + '\\Classification\\dental_cnn_with_size_letterbox.pth'
model = CarrotCNNWithSize().to(DEVICE)
model.load_state_dict(torch.load(save_path, map_location=DEVICE))
model.eval()  # 평가 모드 설정 (Dropout, BatchNorm 등 비활성화)


# 추론
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")  # 흑백, RGBA, BGR 등을 RGB 3채널로 변환해야만 신경망에서 설정한 3채널과 일치함
    orig_w, orig_h = image.size
    area = orig_w * orig_h
    aspect_ratio = orig_w / orig_h    # 이미지 크기 정보 생성 및 GPU에 이동
    size_feat = torch.tensor([[orig_w, orig_h, area, aspect_ratio]], dtype=torch.float32).to(DEVICE)

    # transform 수동 호출
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)   # 이미지 1개를 Batch 형식의 차원으로 변환하여 GPU에 이동
    output = model(image_tensor, size_feat)    # __call__() -> forward() 순전파
    pred = output.argmax(dim=1).item()    # (Batch, classes) 차원의 2번째인 classes에서 가장 큰 값의 인덱스 리턴

    plt.imshow(np.array(image))
    plt.title(f"Prediction: {class_names[pred]}")
    plt.axis('off')
    plt.show()


# 'C:\\Users\\ngins\\Git\\python.initial\\dataset\\bottle\\test\\GOOD\\001.png'
predict_image(data_path+'\\test\\BAD\\001.jpg')  # 실제 파일 경로 지정