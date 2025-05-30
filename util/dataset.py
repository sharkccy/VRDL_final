import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import skimage.feature

class SeaLionPatchDataset:
    def __init__(self, image_files, train_dir, dotted_dir, patch_size=224, scale_factor = 0.4, transform=None):
        self.image_files = image_files
        self.train_dir = train_dir
        self.dotted_dir = dotted_dir
        self.patch_size = patch_size
        self.transform = transform
        self.scale_factor = scale_factor
        self.patches = []
        self.labels = []
        self._prepare_patches()

    def _prepare_patches(self):
        for filename in self.image_files:
            image_1 = cv2.imread(os.path.join(self.dotted_dir, filename))
            image_2 = cv2.imread(os.path.join(self.train_dir, filename))
            if image_1 is None or image_2 is None:
                print(f"Warning: Failed to load images for {filename}")
                continue
            
            h, w = image_1.shape[:2]
            scale_h = int(h * self.scale_factor)
            scale_w = int(w * self.scale_factor)

            image_1_scaled = cv2.resize(image_1, (scale_w, scale_h), interpolation=cv2.INTER_CUBIC)
            image_2_scaled = cv2.resize(image_2, (scale_w, scale_h), interpolation=cv2.INTER_CUBIC)


            img1 = cv2.GaussianBlur(image_1_scaled, (5, 5), 0)
            image_3 = cv2.absdiff(image_1_scaled, image_2_scaled)
            mask_1 = cv2.cvtColor(image_1_scaled, cv2.COLOR_BGR2GRAY)
            mask_1[mask_1 < 50] = 0
            mask_1[mask_1 > 0] = 255
            image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
            image_6 = np.max(image_4, axis=2)
            blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)
            h, w = image_1_scaled.shape[:2]

            for i in range(0, h - self.patch_size + 1, self.patch_size // 2):
                for j in range(0, w - self.patch_size + 1, self.patch_size // 2):
                    patch = image_2_scaled[i:i+self.patch_size, j:j+self.patch_size]
                    if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                        continue
                    counts = np.zeros(5)
                    for blob in blobs:
                        y, x, s = blob
                        if i <= y < i + self.patch_size and j <= x < j + self.patch_size:
                            b, g, r = img1[int(y)][int(x)]
                            if r > 225 and b < 25 and g < 25: counts[0] += 1
                            elif r > 225 and b > 225 and g < 25: counts[1] += 1
                            elif r < 75 and b < 50 and 150 < g < 200: counts[4] += 1
                            elif r < 75 and 150 < b < 200 and g < 75: counts[3] += 1
                            elif 60 < r < 120 and b < 50 and g < 75: counts[2] += 1
                    print(f"Processing {filename} at patch ({i}, {j}): counts = {counts}")
                    self.patches.append((patch, i, j))
                    self.labels.append(counts)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, _, _ = self.patches[idx]
        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        return patch, label