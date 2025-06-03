import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from time import sleep

# 定義路徑（請根據你的環境修改）
train_dir = "Train/"  # 原始訓練圖片資料夾
dotted_dir = "TrainDotted/"  # 原始標註圖片資料夾
train_v2_dir = "Train_v2/"  # 新訓練圖片資料夾
dotted_v2_dir = "TrainDotted_v2/"  # 新標註圖片資料夾
csv_file = "Train/train.csv"  # 計數數據檔案

# 檢查路徑是否存在
if not os.path.exists(train_dir):
    print(f"Error: {train_dir} does not exist.")
    exit()
if not os.path.exists(dotted_dir):
    print(f"Error: {dotted_dir} does not exist.")
    exit()
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} does not exist.")
    exit()

# 檢查 Train 和 TrainDotted 是否包含 .jpg 檔案
train_files = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
dotted_files = [f for f in os.listdir(dotted_dir) if f.endswith('.jpg')]
if not train_files:
    print(f"Warning: No .jpg files found in {train_dir}.")
if not dotted_files:
    print(f"Warning: No .jpg files found in {dotted_dir}.")

# 載入 train.csv
df = pd.read_csv(csv_file)
categories = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']

# 定義目標總和和容許範圍
# min_sum = 2500
# max_sum = 18500
min_sum = 20
max_sum = 100

# 隨機採樣選出平衡子集
def select_balanced_subset():
    best_subset = None
    best_sums = None
    best_std = float('inf')
    best_size = 0
    n_trials = 10000
    min_samples = 1
    max_samples = 800
    np.random.seed(42)

    for _ in range(n_trials):
        n = np.random.randint(min_samples, max_samples + 1)
        subset_ids = np.random.choice(df['train_id'], size=n, replace=False)
        subset = df[df['train_id'].isin(subset_ids)]
        sums = subset[categories].sum().values
        std = np.std(sums)
        if (min_sum <= sums).all() and (sums <= max_sum).all():
            if std < best_std or (std == best_std and n > best_size):
                best_subset = subset_ids
                best_sums = sums
                best_std = std
                best_size = n
    
    return best_subset, best_sums, best_std, best_size

# 選出子集
best_subset, best_sums, best_std, best_size = select_balanced_subset()

if best_subset is None:
    print("No subset found satisfying the constraints.")
    exit()

# 檢查子集中檔案的存在性
valid_subset = []
missing_ids = []
for train_id in best_subset:
    train_id = str(int(train_id))
    print(f"Checking train_id: {train_id}")
    src_train = os.path.join(train_dir, f"{train_id}.jpg")
    src_dotted = os.path.join(dotted_dir, f"{train_id}.jpg")
    if os.path.exists(src_train) and os.path.exists(src_dotted):
        valid_subset.append(train_id)
    else:
        missing_ids.append(train_id)
        if not os.path.exists(src_train):
            print(f"Missing: {src_train}")
        if not os.path.exists(src_dotted):
            print(f"Missing: {src_dotted}")

if not valid_subset:
    print("Error: No valid files found for the selected subset.")
    exit()

if missing_ids:
    print(f"Warning: {len(missing_ids)} train_ids have missing files and were skipped: {missing_ids}")
    best_subset = valid_subset
    best_sums = df[df['train_id'].isin(best_subset)][categories].sum().values
    best_std = np.std(best_sums)
    best_size = len(best_subset)

# 打印選出的子集資訊
print("Selected train_ids:", sorted(best_subset))
print("\nCategory sums:")
for cat, total in zip(categories, best_sums):
    print(f"  {cat}: {total}")
print(f"Standard deviation of sums: {best_std:.2f}")
print(f"Number of samples: {best_size}")

# 建立新資料夾
os.makedirs(train_v2_dir, exist_ok=True)
os.makedirs(dotted_v2_dir, exist_ok=True)

# 複製圖片檔案
missing_files = []
for train_id in best_subset:
    train_id = str(int(train_id))
    ext = '.jpg'
    src_train = os.path.join(train_dir, f"{train_id}{ext}")
    src_dotted = os.path.join(dotted_dir, f"{train_id}{ext}")
    dst_train = os.path.join(train_v2_dir, f"{train_id}{ext}")
    dst_dotted = os.path.join(dotted_v2_dir, f"{train_id}{ext}")
    
    # 檢查並複製 Train 圖片
    if os.path.exists(src_train):
        shutil.copy2(src_train, dst_train)
    else:
        missing_files.append(src_train)
    
    # 檢查並複製 TrainDotted 圖片
    if os.path.exists(src_dotted):
        shutil.copy2(src_dotted, dst_dotted)
    else:
        missing_files.append(src_dotted)

# 報告缺失檔案
if missing_files:
    print("\nMissing files:")
    for f in missing_files:
        print(f"  {f}")
else:
    print("\nAll files copied successfully!")

# 驗證新資料夾的檔案數
train_v2_files = len([f for f in os.listdir(train_v2_dir) if f.endswith('.jpg')])
dotted_v2_files = len([f for f in os.listdir(dotted_v2_dir) if f.endswith('.jpg')])
print(f"\nFiles in Train_v2: {train_v2_files}")
print(f"Files in TrainDotted_v2: {dotted_v2_files}")