import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from net.model_kaggle_ver import SwinRegression
import threading
from queue import Queue

# 解析命令列參數
parser = argparse.ArgumentParser(description="Test Sea Lion Counting Model")
parser.add_argument("--test_dir", type=str, default="E:/Test_split_5", help="Path to test directory")
parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches to extract from images")
parser.add_argument("--num_classes", type=int, default=5, help="Number of classes for regression")
parser.add_argument("--model_path", type=str, default="output/last_mosaic.ckpt", help="Path to the trained model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for patch inference")
args = parser.parse_args()

# 轉換函數
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=0, scale=(0.59, 0.59), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定義 Dataset 處理 Patch
class PatchDataset(Dataset):
    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_pil = Image.fromarray(patch)
        patch_tensor = val_transform(patch_pil)
        return patch_tensor

# 載入模型
model = SwinRegression(num_classes=args.num_classes)
checkpoint = torch.load(args.model_path, map_location='cuda')
state_dict = checkpoint['state_dict']
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to('cuda')
model.eval()

# 推理函數
def predict_patches_swin(model, image_path, patch_size=224, batch_size=32):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    h, w = img.shape[:2]

    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    print(f"Original size: ({h}, {w}), Padded size: ({new_h}, {new_w})")

    pad_h = new_h - h
    pad_w = new_w - w
    padded_img = cv2.copyMakeBorder(
        img, top=0, bottom=pad_h, left=0, right=pad_w,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    patches = []
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = padded_img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            patches.append(patch)

    if not patches:
        print(f"No valid patches extracted for {image_path}")
        return torch.zeros(args.num_classes).cpu().numpy()

    dataset = PatchDataset(patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    counts = torch.zeros(args.num_classes).cuda()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.cuda()
            outputs = model(batch)
            counts += outputs.sum(dim=0)
    print(f"Total counts: {counts.round().int().cpu().numpy()}")
    return counts.round().int().cpu().numpy()

# 執行緒工作函數
def process_images(test_files, test_dir, result_queue):
    results = []
    for test_file in test_files:
        test_id = int(test_file.split('.')[0])
        image_path = os.path.join(test_dir, test_file)
        counts = predict_patches_swin(model, image_path, patch_size=args.patch_size, batch_size=args.batch_size)
        results.append({
            'test_id': test_id,
            'adult_males': counts[0],
            'subadult_males': counts[1],
            'adult_females': counts[2],
            'juveniles': counts[3],
            'pups': counts[4]
        })
    result_queue.put(results)

# 主程式
def main():
    test_dir = args.test_dir
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    # 分割檔案給三個執行緒
    n = len(test_files)
    chunk_size = n // 3
    files_thread1 = test_files[:chunk_size]
    files_thread2 = test_files[chunk_size:2*chunk_size]
    files_thread3 = test_files[2*chunk_size:]

    # 建立結果佇列
    result_queue = Queue()

    # 建立並啟動執行緒
    thread1 = threading.Thread(target=process_images, args=(files_thread1, test_dir, result_queue))
    thread2 = threading.Thread(target=process_images, args=(files_thread2, test_dir, result_queue))
    thread3 = threading.Thread(target=process_images, args=(files_thread3, test_dir, result_queue))
    
    thread1.start()
    thread2.start()
    thread3.start()

    # 等待執行緒完成
    thread1.join()
    thread2.join()
    thread3.join()

    # 合併結果
    submission = []
    for _ in range(3):
        submission += result_queue.get()

    # 儲存結果
    submission_df = pd.DataFrame(submission)
    submission_df = submission_df.sort_values('test_id')
    submission_df.to_csv(f'submission_5_mosaic.csv', index=False)
    print(f"Submission file generated: submission_5_mosaic.csv")

if __name__ == "__main__":
    main()