import os
import cv2
from PIL import Image
import numpy as np
import torch
import pandas as pd
import argparse
import torchvision.transforms as transforms
import timm
from net.model_kaggle_ver import SwinRegression

# 解析命令列參數
parser = argparse.ArgumentParser(description="Test Sea Lion Counting Model")
parser.add_argument("--test_dir", type=str, default="Test", help="Path to test directory")
parser.add_argument("--patch_size", type=int, default=224, help="Size of the patches to extract from images")
parser.add_argument("--num_classes", type=int, default=5, help="Number of classes for regression")
parser.add_argument("--model_path", type=str, default="output/last.ckpt", help="Path to the trained model checkpoint")
args = parser.parse_args()

val_transform = transforms.Compose([
    transforms.ToTensor(),  # 轉為張量 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 標準化
])

# 獲取模型特定的轉換
model = SwinRegression(num_classes=args.num_classes)

# 載入檢查點
checkpoint = torch.load(args.model_path, map_location='cuda')
state_dict = checkpoint['state_dict']
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to('cuda')
model.eval()

# 推理函數
def predict_patches_swin(model, image_path, patch_size=224):
    # 讀取圖像
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 計算填充後的尺寸（最近的 224 倍數）
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    print(f"Original size: ({h}, {w}), Padded size: ({new_h}, {new_w})")

    # 填充圖像到目標尺寸
    pad_h = new_h - h
    pad_w = new_w - w
    padded_img = cv2.copyMakeBorder(
        img,
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # 黑色填充
    )

    counts = torch.zeros(args.num_classes).cuda()
    stride = patch_size  # 無重疊步幅

    for i in range(0, new_h, stride):
        for j in range(0, new_w, stride):
            patch = padded_img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            # 將 patch 轉為浮點數並正規化到 [0, 1]（這裡由 val_transform 處理）
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)  # 轉為 RGB
            patch_pil = Image.fromarray(patch)
            patch_tensor = val_transform(patch_pil).unsqueeze(0).cuda()
            
            with torch.no_grad():
                count = model(patch_tensor)  # 模型輸出 (1, 5)
                # print(f"Patch ({i}, {j}) count: {count.cpu().numpy()}")
            counts += count.squeeze(0)  # 累加到 (5,)
    print(f"Total counts: {counts.cpu().numpy()}")
    return counts.round().int().cpu().numpy()

# 測試
test_dir = args.test_dir
test_files = [f for f in os.listdir(test_dir) if f.endswith('.png') or f.endswith('.jpg')]
submission = []

for test_file in test_files:
    test_id = int(test_file.split('.')[0])
    counts = predict_patches_swin(model, os.path.join(test_dir, test_file), patch_size=args.patch_size)
    submission.append({
        'test_id': test_id,
        'adult_males': counts[0],
        'subadult_males': counts[1],
        'adult_females': counts[2],
        'juveniles': counts[3],
        'pups': counts[4]
    })

submission_df = pd.DataFrame(submission)
submission_df = submission_df.sort_values('test_id')
submission_df.to_csv('submission_swin_pytorch.csv', index=False)
print("Submission file generated: submission_swin_pytorch.csv")