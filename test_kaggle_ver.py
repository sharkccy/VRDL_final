import os
import cv2
import numpy as np
import torch
import pandas as pd
import argparse
import torchvision.transforms as transforms
from net.model_kaggle_ver import SwinRegression

# 解析命令列參數
parser = argparse.ArgumentParser(description="Test Sea Lion Counting Model")
parser.add_argument("--test_dir", type=str, default="Test", help="Path to test directory")
args = parser.parse_args()

# 數據增強（僅用於調整尺寸）
val_transform = transforms.Compose([
    transforms.Resize((224, 224))
])

# 推理函數
def predict_patches_swin(model, image_path, patch_size=224):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    counts = torch.zeros(5).cuda()
    stride = patch_size
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            patch = patch.astype(np.float32) / 255.0
            patch = np.transpose(patch, (2, 0, 1))
            patch = torch.from_numpy(patch).unsqueeze(0).cuda()
            patch = val_transform(patch)
            with torch.no_grad():
                count = model(patch)  
            counts += count.squeeze()

    return counts.round().int().cpu().numpy()

# 測試
test_dir = args.test_dir
test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
submission = []

model = SwinRegression(num_classes=5).cuda()
model.load_state_dict(torch.load('work_dirs/swin_pytorch_best.pth'))
model.eval()

for test_file in test_files:
    test_id = int(test_file.split('.')[0])
    counts = predict_patches_swin(model, os.path.join(test_dir, test_file))
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