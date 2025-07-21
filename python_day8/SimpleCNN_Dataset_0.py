# 사용자 정의 Dataset 클래스 추가
import os
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = label_map  # 예: {'cat': 0, 'dog': 1}

        for class_name, label in label_map.items():
            class_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx): # DataLoader가 호출(직접호출하지 않음), 데이터 1set(문제, 정답)를 조회할 때
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label #가로, 세로, 넓이 ... 등을 추가하여 더  많은 정보를 전달한다.
