import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.models as models
import timm
import numpy as np
import pandas as pd
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, test_dir, transform = None):
        self.test_dir = test_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# 資料路徑
test_dir = 'data/test'
train_dir = 'data/train'
model_name = 'model.pth'
model_type = 'vit_base_patch16_224'  # 更改為 vit_base 模型

# 超參數
num_classes = 5
batch_size = 128

# 檢查是否有 GPU 可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 資料預處理
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize((224, 224), interpolation=3),  # ViT 模型需要輸入 224x224 大小的圖片
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# 載入資料
train_dataset = torchvision.datasets.ImageFolder(
    root=train_dir,
    transform=transform
)

test_dataset = CustomDataset(test_dir=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 載入模型
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)

checkpoint = torch.load(model_name, map_location=device)
state_dict = checkpoint['model_state_dict']
if torch.cuda.device_count() == 1 and list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
elif torch.cuda.device_count() > 1 and not list(state_dict.keys())[0].startswith('module.'):
    state_dict = {'module.' + k: v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.to(device)
print("模型已載入，用於推論")
del checkpoint
del state_dict

# 推論
predictions_by_test_id = {}

model.eval()
with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for filename, class_index in zip(img_names, predicted):
            print(f"filename: {filename}, class_index: {class_index}")
            # 從檔名中提取 test_id，例如 "123_1.jpg" -> test_id = 123
            test_id = int(filename.split('_')[0])
            predicted_class = train_dataset.classes[class_index]

            # 如果 test_id 尚未記錄，則初始化計數
            if test_id not in predictions_by_test_id:
                predictions_by_test_id[test_id] = {
                    'adult_males': 0,    # class_0
                    'subadult_males': 0, # class_1
                    'adult_females': 0,  # class_2
                    'juveniles': 0,      # class_3
                    'pups': 0            # class_4
                }

            # 將預測的類別映射到對應的欄位
            if predicted_class == 'class_0':
                predictions_by_test_id[test_id]['adult_males'] += 1
            elif predicted_class == 'class_1':
                predictions_by_test_id[test_id]['subadult_males'] += 1
            elif predicted_class == 'class_2':
                predictions_by_test_id[test_id]['adult_females'] += 1
            elif predicted_class == 'class_3':
                predictions_by_test_id[test_id]['juveniles'] += 1
            elif predicted_class == 'class_4':
                predictions_by_test_id[test_id]['pups'] += 1
            # 超過 class_4 的類別不計入結果

# 準備 CSV 資料
test_ids = sorted(predictions_by_test_id.keys())
data = {
    'test_id': test_ids,
    'adult_males': [predictions_by_test_id[tid]['adult_males'] for tid in test_ids],
    'subadult_males': [predictions_by_test_id[tid]['subadult_males'] for tid in test_ids],
    'adult_females': [predictions_by_test_id[tid]['adult_females'] for tid in test_ids],
    'juveniles': [predictions_by_test_id[tid]['juveniles'] for tid in test_ids],
    'pups': [predictions_by_test_id[tid]['pups'] for tid in test_ids]
}

# 建立 DataFrame 並儲存為 CSV
result = pd.DataFrame(data)
result.to_csv('submission.csv_75', index=False)
print("submission.csv_75 已儲存！")