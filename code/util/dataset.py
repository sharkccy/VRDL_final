import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import skimage.feature
import time
import random
from multiprocessing import Pool, cpu_count

class SeaLionPatchDataset:
    def __init__(self, image_files, train_dir, dotted_dir, patch_size=256, scale_factor=0.4, transform=None, num_workers=None):
        self.image_files = image_files
        self.train_dir = train_dir
        self.dotted_dir = dotted_dir
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.transform = transform
        self.num_workers = 8
        self.patches = []
        self.labels = []
        self._prepare_patches()

    def _process_single_image(self, filename):
        try:
            image_1 = cv2.imread(os.path.join(self.dotted_dir, filename))
            image_2 = cv2.imread(os.path.join(self.train_dir, filename))
            if image_1 is None or image_2 is None:
                print(f"Warning: Failed to load images for {filename}")
                return [], []

            # 在原始圖像上進行 blob 檢測
            image_1 = cv2.GaussianBlur(image_1, (5, 5), 0)
            image_3 = cv2.absdiff(image_1, image_2)
            mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
            mask_1[mask_1 < 50] = 0
            mask_1[mask_1 > 0] = 255
            image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
            image_6 = np.max(image_4, axis=2)
            blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

            # 計算縮放後的尺寸
            h, w = image_1.shape[:2]
            scale_h = int(h * self.scale_factor)
            scale_w = int(w * self.scale_factor)

            # 應用背景遮罩
            ma = cv2.cvtColor((1 * (np.sum(image_1, axis=2) > 20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
            image_2_masked = image_2 * ma

            # 將 blob 坐標映射到縮放後的坐標系
            blobs_scaled = [(int(y * self.scale_factor), int(x * self.scale_factor), s) for y, x, s in blobs]

            # 計算網格尺寸
            num_patches_x = int(scale_w // self.patch_size) + 1
            num_patches_y = int(scale_h // self.patch_size) + 1
            res = np.zeros((num_patches_x, num_patches_y, 5), dtype='int16')

            # 計算每個網格的計數
            for blob in blobs_scaled:
                y, x, s = blob
                x1 = int(x // self.patch_size)
                y1 = int(y // self.patch_size)
                if 0 <= x1 < num_patches_x and 0 <= y1 < num_patches_y:
                    scaled_y = int(y / self.scale_factor)
                    scaled_x = int(x / self.scale_factor)
                    b, g, r = image_1[scaled_y][scaled_x]
                    if r > 225 and b < 25 and g < 25:   # RED
                        res[x1, y1, 0] += 1
                    elif r > 225 and b > 225 and g < 25:    # MAGENTA
                        res[x1, y1, 1] += 1
                    elif r < 75 and b < 50 and 150 < g < 200: # GREEN
                        res[x1, y1, 4] += 1
                    elif r < 75 and 150 < b < 200 and g < 75: # BLUE
                        res[x1, y1, 3] += 1
                    elif 60 < r < 120 and b < 50 and g < 75: # BROWN
                        res[x1, y1, 2] += 1

            patches = []
            labels = []
            
            # 提取補丁
            for i in range(num_patches_x):
                for j in range(num_patches_y):
                    x_start = int((i * self.patch_size) / self.scale_factor)
                    y_start = int((j * self.patch_size) / self.scale_factor)
                    if x_start + self.patch_size <= w and y_start + self.patch_size <= h:
                        # 如果這個 patch 沒有海獅，以 0.75 的機率跳過
                        if np.sum(res[i, j, :]) == 0 and np.random.random() < 0.75:
                            continue
                        train_patch = image_2_masked[y_start:y_start + self.patch_size, x_start:x_start + self.patch_size]
                        patches.append((train_patch, y_start, x_start))
                        labels.append(res[i, j, :])
                        print(f"Processing {filename} at patch ({y_start}, {x_start}): counts = {res[i, j, :]}")

            return patches, labels
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return [], []

    def _prepare_patches(self):
        print(f"Starting multiprocessing with {self.num_workers} workers...")
        start_time = time.time()

        # 使用進程池處理圖片
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_single_image, self.image_files)

        # 合併所有結果
        all_patches = []
        all_labels = []
        for patches, labels in results:
            all_patches.extend(patches)
            all_labels.extend(labels)

        # 將標籤轉換為 numpy array 以便更快的處理
        all_labels_array = np.array(all_labels)
        
        # 找出正樣本和背景樣本的索引
        positive_indices = np.where(np.sum(all_labels_array, axis=1) > 0)[0]
        background_indices = np.where(np.sum(all_labels_array, axis=1) == 0)[0]
        
        # 計算需要的背景樣本數量
        num_positive = len(positive_indices)
        num_background_needed = min(len(background_indices), num_positive * 3)

        # 隨機選擇背景樣本
        np.random.seed(int(time.time()) % 1000)
        selected_background_indices = np.random.choice(background_indices, num_background_needed, replace=False)

        # 合併正樣本和選定的背景樣本的索引
        selected_indices = np.concatenate([positive_indices, selected_background_indices])
        np.random.shuffle(selected_indices)

        # 使用選定的索引提取樣本
        self.patches = [all_patches[i] for i in selected_indices]
        self.labels = [all_labels[i] for i in selected_indices]

        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        print(f"Total patches: {len(self.patches)}, Positive: {num_positive}, Background: {num_background_needed}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, _, _ = self.patches[idx]
        label = self.labels[idx]
        if self.transform:
            patch = self.transform(patch)
        # print(f"Patch shape after transform: {patch.shape}")  # 應為 (3, 224, 224)
        label = torch.from_numpy(label).float()
        return patch, label