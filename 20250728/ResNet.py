#########################################
#########################################


# #########################################
# # ResNet 불러오기
# #########################################
# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision.models import resnet18, ResNet18_Weights

# def get_resnet_model(num_classes):
#     weights = ResNet18_Weights.DEFAULT  # 최신 가중치
#     model = resnet18(weights=weights)

#     # 모든 레이어를 freeze (학습 안함)
#     for param in model.parameters():
#         param.requires_grad = False

#     for name, param in model.named_parameters():
#         if "layer3" in name or "layer4" in name or "fc" in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False

#     # Dropout 적용 예시
#     in_features = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Dropout(p=0.5),
#         nn.Linear(in_features, num_classes)
#     )

#     return model

# #########################################
# # 학습 루프 및 평가
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from tqdm import tqdm

# import matplotlib.pyplot as plt

# def train_model(model, train_loader, val_loader, device, epochs=5):
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     #optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     # AdamW : #torch.optim.AdamW는 PyTorch에서 제공하는 Adam의 개선 버전 (모델의 일반화 성능을 향)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # epoch 20 안에서 cos 함수모양으로 수렴
#     # 코사인 함수 형태로 점진적으로 감소시키는 스케줄러
#     # 학습이 진행됨에 따라 Learning rate를 부드럽게 줄여서 최적화 성능을 높이기 위한 전략
#     # T_max : 총 코사인 주기 (스케줄러가 완료되는 epoch 수)

#     # ⬇️ 시각화를 위한 저장 리스트
#     train_loss_list = []
#     val_acc_list = []

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0

#         for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         train_loss_list.append(avg_loss)

#         val_acc = evaluate_model(model, val_loader, device, verbose=True)
#         val_acc_list.append(val_acc)

#         print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

#     # ⬇️ 학습 후 시각화
#     plot_training(train_loss_list, val_acc_list)


# def evaluate_model(model, val_loader, device, verbose=False):
#     model.eval()
#     correct = total = 0

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             preds = torch.argmax(outputs, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     acc = 100 * correct / total
#     if verbose:
#         print(f"Validation Accuracy: {acc:.2f}%")
#     return acc


#########################################
# Dataset 클래스(YOLO규격의 데이터)
#########################################
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import timm
from torchvision import transforms
import os
from torch.utils.data import Dataset
from PIL import Image


class YoloStyleClassificationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = [f for f in os.listdir(
            image_dir) if f.endswith(('.jpg', '.png'))]
        self.image_files.sort()

        # 클래스 ID 수집
        class_ids = set()
        for img_file in self.image_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_ids.add(int(line.strip().split()[0]))
        self.classes = sorted(list(class_ids))
        self.num_classes = len(self.classes)
        self.class_to_idx = {cid: idx for idx, cid in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)

        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_filename)

        image = Image.open(image_path).convert("RGB")

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                first_line = f.readline().strip()
                class_id = int(first_line.split()[0]) if first_line else 0
        else:
            class_id = 0

        label = self.class_to_idx.get(class_id, 0)

        if self.transform:
            image = self.transform(image)

        return image, label


#########################################
# 시각화 함수
def plot_training(loss_list, acc_list):
    epochs = range(1, len(loss_list)+1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_list, 'b-o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_list, 'g-o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.show()


#########################################
# Transform (EfficientNet 규격 + Letterbox)
#########################################


class Letterbox:
    def __init__(self, target_size=(224, 224), fill_color=(114, 114, 114)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, image):
        iw, ih = image.size
        w, h = self.target_size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BILINEAR)
        new_image = Image.new('RGB', self.target_size, self.fill_color)
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image


transform = transforms.Compose([
    Letterbox((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


#########################################
# EfficientNet-B0 모델 불러오기 (timm 사용)
#########################################


def get_efficientnet_model(num_classes):
    model = timm.create_model('efficientnet_b0', pretrained=True)
    in_features = model.classifier.in_features
    # 교체하면 디폴트 requires_grad=True 상태
    model.classifier = nn.Linear(in_features, num_classes)

    # 전체 freeze
    '''
    for param in model.parameters():
        param.requires_grad = False

    # 마지막 두 블록과 classifier만 학습
    '''
    '''
    for name, param in model.named_parameters():
        if 'blocks.6' in name or 'blocks.7' in name or 'classifier' in name:
            param.requires_grad = True
    '''

    # classifier만 학습 허용
    '''
    for param in model.classifier.parameters():
        param.requires_grad = True
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    '''
    # classifier(FC네트워크)를 교체하는 경우 예시
    '''
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    '''
    return model


#########################################
# 학습 및 평가 루프
#########################################


def evaluate_model(model, val_loader, device, verbose=False):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    if verbose:
        print(f"Validation Accuracy: {acc:.2f}%")
    return acc


def plot_training(loss_list, acc_list):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Train Loss')
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(acc_list, label='Val Acc')
    plt.title("Accuracy")
    plt.show()


def train_model(model, train_loader, val_loader, device, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_accuracies = []
    best_acc = 0.0
    best_train_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # if (total_loss / len(train_loader)) > 0.015:
            #     print(f"Epoch {epoch+1} - Loss: {(total_loss / len(train_loader)):.4f} | images: {images}") # tensor

        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_train_loss = total_loss / len(train_loader)
        val_acc = evaluate_model(model, val_loader, device, verbose=True)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # earlyStopping: 가장 좋은 수치의 모델이 나오면 그 것만 저장하고...
        if val_acc > best_acc:
            best_acc = val_acc
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), "best_efficient_model.pth")
            print(f"✅ New best val_acc: {val_acc:.2f}% → Model saved.")

        # 정확도가 동일하다면, 학습(손실)오차가 적은 경우가 일반화 성능이 일반적으로 높다
        elif val_acc == best_acc and avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), "best_efficient_model.pth")
            print(f"✅ Same val_acc but lower train loss → Model saved.")

        model.load_state_dict(torch.load("best_efficient_model.pth"))

    plot_training(train_losses, val_accuracies)


#########################################
# 전체 실행
#########################################
dataset_path = "C:/Users/ngins/Git/python.initial/dataset/bottle.yolov11"  # 데이터셋 경로

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = YoloStyleClassificationDataset(
        image_dir=dataset_path+"/train/images",
        label_dir=dataset_path+"/train/labels",
        transform=transform
    )

    val_dataset = YoloStyleClassificationDataset(
        image_dir=dataset_path+"/valid/images",
        label_dir=dataset_path+"/valid/labels",
        transform=transform
    )
    '''
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    '''
    train_loader = DataLoader(train_dataset, batch_size=32,
                                shuffle=True, drop_last=True)  # 마지막 배치는 32가 안될 수 있으므로 배제
    val_loader = DataLoader(val_dataset, batch_size=32,
                            drop_last=False)  # 검증은 유지 가능
    model = get_efficientnet_model(num_classes=train_dataset.num_classes)

    train_model(model, train_loader, val_loader, device, epochs=100)
