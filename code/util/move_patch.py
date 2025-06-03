import os
import shutil
from pathlib import Path
import re

def organize_patches(input_dir="E:/Test_tile"):
    """
    將 temp_output 中的 patch 圖片按 image_id 整理到子資料夾。
    輸入檔名格式: {image_id}_tile_{row}_{col}_{size}x{size}.jpg
    輸出結構: temp_output/{image_id}/tile_{row}_{col}.jpg
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"目錄 {input_dir} 不存在")
        return

    # 掃描所有 .jpg 檔案
    patch_files = list(input_dir.glob("*.jpg"))
    if not patch_files:
        print(f"在 {input_dir} 中未找到任何 .jpg 檔案")
        return

    # 正則表達式匹配檔名
    pattern = re.compile(r"(\d+)_tile_(\d+)_(\d+)_224x224\.jpg")

    for patch_file in patch_files:
        match = pattern.match(patch_file.name)
        if not match:
            print(f"檔名格式不符，跳過: {patch_file}")
            continue

        image_id, row, col = match.groups()
        # 創建子資料夾 temp_output/{image_id}
        sub_dir = input_dir / image_id
        sub_dir.mkdir(exist_ok=True)

        # 新檔名: tile_{row}_{col}.jpg
        new_filename = f"tile_{row}_{col}.jpg"
        dest_path = sub_dir / new_filename

        # 移動檔案
        try:
            shutil.move(str(patch_file), str(dest_path))
            print(f"已移動: {patch_file} -> {dest_path}")
        except Exception as e:
            print(f"移動失敗 {patch_file}: {e}")

    print("Patch 整理完成！")

if __name__ == "__main__":
    organize_patches()