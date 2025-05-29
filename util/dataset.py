import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import skimage.feature
import cv2

def GetData(filename, train_dir, dotted_dir, patch_size=112, overlap=56, downsample_ratio=4):
    image_1 = cv2.imread(os.path.join(dotted_dir, filename))
    image_2 = cv2.imread(os.path.join(train_dir, filename))
    img1 = cv2.GaussianBlur(image_1, (5, 5), 0)

    image_3 = cv2.absdiff(image_1, image_2)
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    image_6 = np.max(image_4, axis=2)
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

    points = []
    for blob in blobs:
        y, x, s = blob
        b, g, R = img1[int(y)][int(x)][:]
        if R > 225 and b < 25 and g < 25:
            cls = 0
        elif R > 225 and b > 225 and g < 25:
            cls = 1
        elif R < 75 and b < 50 and 150 < g < 200:
            cls = 4
        elif R < 75 and 150 < b < 200 and g < 75:
            cls = 3
        elif 60 < R < 120 and b < 50 and g < 75:
            cls = 2
        else:
            continue
        points.append((x, y, cls))

    h, w = image_2.shape[:2]
    patches = []
    heatmaps = []
    offsets = []
    heatmap_size = (patch_size // downsample_ratio, patch_size // downsample_ratio)

    for i in range(0, h - patch_size + 1, patch_size - overlap):
        for j in range(0, w - patch_size + 1, patch_size - overlap):
            patch = image_2[i:i+patch_size, j:j+patch_size]
            patch_points = [(x-j, y-i, cls) for x, y, cls in points 
                           if j <= x < j + patch_size and i <= y < i + patch_size]
            heatmap, offset = generate_heatmap(patch_points, heatmap_size, num_classes=5, 
                                              sigma=2, downsample_ratio=downsample_ratio)
            patches.append(patch)
            heatmaps.append(heatmap)
            offsets.append(offset)

    return patches, heatmaps, offsets

def generate_heatmap(points, heatmap_size=(28, 28), num_classes=5, sigma=2, downsample_ratio=4):
    heatmap = np.zeros((heatmap_size[0], heatmap_size[1], num_classes), dtype=np.float32)
    offset = np.zeros((heatmap_size[0], heatmap_size[1], 2), dtype=np.float32)

    for x, y, cls in points:
        x_down = int(x / downsample_ratio)
        y_down = int(y / downsample_ratio)
        if 0 <= x_down < heatmap_size[1] and 0 <= y_down < heatmap_size[0]:
            offset[y_down, x_down] = [x / downsample_ratio - x_down, y / downsample_ratio - y_down]
            for i in range(heatmap_size[0]):
                for j in range(heatmap_size[1]):
                    dist = (i - y_down) ** 2 + (j - x_down) ** 2
                    value = np.exp(-dist / (2 * sigma ** 2))
                    heatmap[i, j, cls] = max(heatmap[i, j, cls], value)

    return heatmap, offset

class SeaLionDataset(Dataset):
    def __init__(self, train_files, train_dir, dotted_dir, patch_size=112, overlap=56, downsample_ratio=4):
        self.train_files = train_files
        self.train_dir = train_dir
        self.dotted_dir = dotted_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.downsample_ratio = downsample_ratio
        self.patches = []
        self.heatmaps = []
        self.offsets = []
        self._prepare_patches()

    def _prepare_patches(self):
        for filename in self.train_files:
            patches, heatmaps, offsets = GetData(filename, self.train_dir, self.dotted_dir, 
                                                self.patch_size, self.overlap, self.downsample_ratio)
            self.patches.extend(patches)
            self.heatmaps.extend(heatmaps)
            self.offsets.extend(offsets)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx].astype(np.float32) / 255.0
        patch = torch.from_numpy(patch.transpose(2, 0, 1))
        heatmap = torch.from_numpy(self.heatmaps[idx])
        offset = torch.from_numpy(self.offsets[idx])
        return patch, heatmap, offset