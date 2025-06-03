import os
import shutil
import math
from pathlib import Path

def split_test_data(source_dir='E:/Test', num_splits=6):
    # 確保源目錄存在
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"找不到源目錄：{source_dir}")

    # 獲取所有PNG檔案並排序
    png_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.jpg')], 
                      key=lambda x: int(x.split('.')[0]))
    
    total_files = len(png_files)
    files_per_split = math.ceil(total_files / num_splits)
    
    print(f"總共找到 {total_files} 個檔案")
    print(f"每份約含 {files_per_split} 個檔案")

    # 創建分割後的目錄
    for i in range(num_splits):
        split_dir = f"E:/Test_split_{i}"
        os.makedirs(split_dir, exist_ok=True)
        print(f"創建目錄：{split_dir}")

        # 計算當前分割的起始和結束索引
        start_idx = i * files_per_split
        end_idx = min((i + 1) * files_per_split, total_files)
        
        # 複製檔案到對應的分割目錄
        for j in range(start_idx, end_idx):
            src_file = os.path.join(source_dir, png_files[j])
            dst_file = os.path.join(split_dir, png_files[j])
            shutil.copy2(src_file, dst_file)
        
        print(f"Test_split_{i} 包含檔案：{png_files[start_idx].split('.')[0]} 到 {png_files[end_idx-1].split('.')[0]}")

if __name__ == "__main__":
    try:
        split_test_data()
        print("\n資料分割完成！")
    except Exception as e:
        print(f"發生錯誤：{str(e)}") 