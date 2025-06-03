import os
import cv2
import numpy as np

def split_image(image, output_dir, base_name):
    h, w = image.shape[:2]
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(0, h, 800):
        for j in range(0, w, 800):
            # 計算當前塊的尺寸
            crop_h = min(800, h - i)
            crop_w = min(800, w - j)
            if crop_h <= 0 or crop_w <= 0:
                continue
            
            # 裁剪圖像
            crop = image[i:i + crop_h, j:j + crop_w]
            
            # 保存裁剪後的圖像
            output_path = os.path.join(output_dir, f"{base_name}_{i // 800}_{j // 800}.png")
            cv2.imwrite(output_path, crop)
            print(f"Saved cropped image to {output_path}, size: {crop_w}x{crop_h}")

def process_bboxes(train_folder, traindotted_folder, output_train_folder, output_traindotted_folder):
    # 處理 train 資料夾
    for img_name in os.listdir(train_folder):
        if img_name.endswith('.png'):
            img_path = os.path.join(train_folder, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                h, w = image.shape[:2]
                if w > 800 or h > 800:
                    base_name = os.path.splitext(img_name)[0]
                    split_image(image, output_train_folder, base_name)
                else:
                    # 如果小於等於 800x800，直接複製
                    output_path = os.path.join(output_train_folder, img_name)
                    cv2.imwrite(output_path, image)
                    print(f"Copied image to {output_path}, size: {w}x{h}")
    
    # 處理 traindotted 資料夾
    for img_name in os.listdir(traindotted_folder):
        if img_name.endswith('.png'):
            img_path = os.path.join(traindotted_folder, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                h, w = image.shape[:2]
                if w > 800 or h > 800:
                    base_name = os.path.splitext(img_name)[0]
                    split_image(image, output_traindotted_folder, base_name)
                else:
                    # 如果小於等於 800x800，直接複製
                    output_path = os.path.join(output_traindotted_folder, img_name)
                    cv2.imwrite(output_path, image)
                    print(f"Copied image to {output_path}, size: {w}x{h}")

if __name__ == "__main__":
    train_folder = "OutputBboxes/merged_bboxes/train"
    traindotted_folder = "OutputBboxes/merged_bboxes/traindotted"
    output_train_folder = "OutputBboxes/split_bboxes/train"
    output_traindotted_folder = "OutputBboxes/split_bboxes/traindotted"
    
    process_bboxes(train_folder, traindotted_folder, output_train_folder, output_traindotted_folder)