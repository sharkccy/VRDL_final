import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import skimage.feature
from sklearn.model_selection import KFold
from mmdet.apis import init_detector, inference_detector, train_detector
from mmcv import Config
import pandas as pd
from util.dataset import SeaLionDataset

train_dir = "Train"
dotted_dir = "TrainDotted"
train_files = [f for f in os.listdir(train_dir) if f.endswith('.png')]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cfg = Config.fromdict(dict(
    _base_='mmdet::centernet/centernet_swin-t-p4-w7_fpn_ms-3x_coco.py',
    model=dict(
        backbone=dict(
            type='SwinTransformer',
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            pretrained='',
        ),
        bbox_head=dict(
            num_classes=5,
        ),
    ),
    data=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
    ),
    optimizer=dict(type='AdamW', lr=0.0001),
    runner=dict(type='EpochBasedRunner', max_epochs=50),
))

fold = 0
for train_idx, val_idx in kf.split(train_files):
    print(f"Training Fold {fold + 1}/5")
    train_subset = [train_files[i] for i in train_idx]
    val_subset = [train_files[i] for i in val_idx]

    train_dataset = SeaLionDataset(train_subset, train_dir, dotted_dir, patch_size=112, overlap=56)
    val_dataset = SeaLionDataset(val_subset, train_dir, dotted_dir, patch_size=112, overlap=56)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    model = init_detector(cfg, device='cuda:0')
    train_detector(
        model,
        train_loader,
        cfg,
        val_dataset=val_loader,
        work_dir=f'work_dirs/fold_{fold}',
    )

    torch.save(model.state_dict(), f'work_dirs/fold_{fold}/best.pth')
    fold += 1

def predict_patches(model, image_path, patch_size=112, overlap=56, downsample_ratio=4):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    all_points = []

    for i in range(0, h - patch_size + 1, patch_size - overlap):
        for j in range(0, w - patch_size + 1, patch_size - overlap):
            patch = img[i:i+patch_size, j:j+patch_size]
            patch = patch.astype(np.float32) / 255.0
            patch = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).cuda()

            result = inference_detector(model, patch)
            for cls, bboxes in enumerate(result):
                for bbox in bboxes:
                    if len(bbox) > 0:
                        x, y = bbox[:2]
                        all_points.append((j + x * downsample_ratio, i + y * downsample_ratio, cls))

    unique_points = []
    for point in all_points:
        x, y, cls = point
        too_close = False
        for up in unique_points:
            ux, uy, ucls = up
            if ucls == cls and ((x - ux) ** 2 + (y - uy) ** 2) ** 0.5 < 5:
                too_close = True
                break
        if not too_close:
            unique_points.append(point)

    counts = {i: 0 for i in range(5)}
    for _, _, cls in unique_points:
        counts[cls] += 1
    return counts

test_dir = "Test"
test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
submission = []

model = init_detector(cfg, 'work_dirs/fold_0/best.pth', device='cuda:0')

for test_file in test_files:
    test_id = int(test_file.split('.')[0])
    counts = predict_patches(model, os.path.join(test_dir, test_file), patch_size=112, overlap=56)
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
submission_df.to_csv('submission.csv', index=False)
print("Submission file generated: submission.csv")