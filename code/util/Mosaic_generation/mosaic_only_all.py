import os
import cv2
import numpy as np
from collections import namedtuple
from copy import deepcopy
import random

# Define Rectangle namedtuple
Rectangle = namedtuple('Rectangle', ['x', 'y', 'w', 'h'])

def phsppog(width, rectangles, sorting="width"):
    """
    The PH heuristic for the Strip Packing Problem (OG variant, no rotations, guillotine constraint).
    """
    if sorting not in ["width", "height"]:
        raise ValueError("The algorithm only supports sorting by width or height but {} was given.".format(sorting))
    if sorting == "width":
        wh = 0
    else:
        wh = 1
    result = [None] * len(rectangles)
    remaining = deepcopy(rectangles)
    sorted_indices = sorted(range(len(remaining)), key=lambda x: -remaining[x][wh])
    sorted_rect = [remaining[idx] for idx in sorted_indices]
    x, y, w, h, H = 0, 0, 0, 0, 0
    while sorted_indices:
        idx = sorted_indices.pop(0)
        r = remaining[idx]
        result[idx] = Rectangle(x, y, r[0], r[1])
        x, y, w, h, H = r[0], H, width - r[0], r[1], H + r[1]
        recursive_packing(x, y, w, h, 0, remaining, sorted_indices, result)
        x, y = 0, H
    return H, result

def recursive_packing(x, y, w, h, D, remaining, indices, result):
    """Helper function to recursively fit a certain area."""
    priority = 6
    for idx in indices:
        for j in range(0, D + 1):
            if priority > 1 and remaining[idx][(0 + j) % 2] == w and remaining[idx][(1 + j) % 2] == h:
                priority, orientation, best = 1, j, idx
                break
            elif priority > 2 and remaining[idx][(0 + j) % 2] == w and remaining[idx][(1 + j) % 2] < h:
                priority, orientation, best = 2, j, idx
            elif priority > 3 and remaining[idx][(0 + j) % 2] < w and remaining[idx][(1 + j) % 2] == h:
                priority, orientation, best = 3, j, idx
            elif priority > 4 and remaining[idx][(0 + j) % 2] < w and remaining[idx][(1 + j) % 2] < h:
                priority, orientation, best = 4, j, idx
            elif priority > 5:
                priority, orientation, best = 5, j, idx
    if priority < 5:
        if orientation == 0:
            omega, d = remaining[best][0], remaining[best][1]
        else:
            omega, d = remaining[best][1], remaining[best][0]
        result[best] = Rectangle(x, y, omega, d)
        indices.remove(best)
        if priority == 2:
            recursive_packing(x, y + d, w, h - d, D, remaining, indices, result)
        elif priority == 3:
            recursive_packing(x + omega, y, w - omega, h, D, remaining, indices, result)
        elif priority == 4:
            if not indices:  # 檢查 indices 是否為空，若為空則提前返回
                return
            min_w = min(min(remaining[idx][0] for idx in indices), min(remaining[idx][1] for idx in indices))
            if w - omega < min_w:
                recursive_packing(x, y + d, w, h - d, D, remaining, indices, result)
            elif h - d < min_w:
                recursive_packing(x + omega, y, w - omega, h, D, remaining, indices, result)
            elif omega < min_w:
                recursive_packing(x + omega, y, w - omega, d, D, remaining, indices, result)
                recursive_packing(x, y + d, w, h - d, D, remaining, indices, result)
            else:
                recursive_packing(x, y + d, omega, h - d, D, remaining, indices, result)
                recursive_packing(x + omega, y, w - omega, h, D, remaining, indices, result)

def create_mosaic(train_folder, traindotted_folder, background_folder, output_root):
    # 讀取所有圖片路徑，只篩選 .png 檔案
    train_images = sorted([f for f in os.listdir(train_folder) if f.endswith('.png')])
    traindotted_images = sorted([f for f in os.listdir(traindotted_folder) if f.endswith('.png')])
    background_images = sorted([f for f in os.listdir(background_folder) if f.endswith('.png')])

    # 輸出讀到的檔案名稱，方便檢查
    print("Train images loaded:")
    for img in train_images:
        print(f"  {img}")
    print("TrainDotted images loaded:")
    for img in traindotted_images:
        print(f"  {img}")

    # 輸出讀到的 merged bbox 數量
    num_merged_bboxes = len(train_images)
    print(f"Loaded {num_merged_bboxes} merged bbox images from {train_folder} and {traindotted_folder}")
    num_background_images = len(background_images)
    print(f"Loaded {num_background_images} background images from {background_folder}")

    if not train_images or not traindotted_images:
        print("No images found in train or traindotted folders")
        return
    if not background_images:
        print("No background images found in background folder")
        return

    # 檢查圖片名稱是否匹配，由於名稱完全一致，直接比較
    paired_images = []
    train_set = set(train_images)
    for traindotted_img in traindotted_images:
        if traindotted_img in train_set:
            paired_images.append((traindotted_img, traindotted_img))  # 配對相同名稱
    if len(paired_images) != len(train_images):
        print("Mismatch in number of images between train and traindotted folders")
        return

    # 更新 train_images 和 traindotted_images 為配對後的檔案
    train_images, traindotted_images = zip(*paired_images)
    train_images = list(train_images)
    traindotted_images = list(traindotted_images)
    print(f"After pairing, {len(train_images)} images will be processed")

    # 提取 bbox 資訊 (使用圖片長寬作為 bbox 寬高)
    bbox_list = []
    for img_name in train_images:
        img_path = os.path.join(train_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 讀取 PNG 圖片並保留透明通道
        if img is not None:
            h, w = img.shape[:2]
            bbox_list.append([0, 0, w, h])  # [x1, y1, x2, y2]，x1, y1 設為 0
            print(f"Extracted bbox for {img_name}: width={w}, height={h}")
        else:
            print(f"Failed to load image for bbox extraction: {img_path}")

    if not bbox_list:
        print("No valid bbox data extracted")
        return

    # 準備 packing 用的寬高列表並分組
    boxes = [[b[2] - b[0], b[3] - b[1]] for b in bbox_list]  # 提取寬高
    large_boxes_with_index = []  # 寬或高超過 400 的圖片
    small_boxes_with_index = []  # 寬和高均小於等於 400 的圖片
    invalid_boxes = []  # 大於 1000x1000 的圖片

    for idx, (w, h) in enumerate(boxes):
        area = w * h
        if w > 1000 or h > 1000:  # 更新為 1000x1000
            invalid_boxes.append((area, idx))
        elif w > 400 or h > 400:
            large_boxes_with_index.append((area, idx))
        else:
            small_boxes_with_index.append((area, idx))

    # 按面積排序
    large_boxes_with_index.sort(reverse=True)  # 大圖片按面積降序
    small_boxes_with_index.sort()  # 小圖片按面積升序

    remaining_large = [item[1] for item in large_boxes_with_index]
    remaining_small = [item[1] for item in small_boxes_with_index]
    total_area_threshold = 1000 * 1000 * 0.8  # 更新總面積閾值為 1000x1000 (1,000,000 像素)
    mosaic_idx = 0

    # 處理大於 400x400 的圖片，獨立生成
    for _, large_idx in large_boxes_with_index:
        # 隨機選擇一張背景圖
        background_path = os.path.join(background_folder, random.choice(background_images))
        background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
        if background is None:
            print(f"Failed to load background image: {background_path}")
            background = np.zeros((1000, 1000, 4), dtype=np.uint8)  # 更新為 1000x1000
        elif background.shape[:2] != (1000, 1000):
            background = cv2.resize(background, (1000, 1000), interpolation=cv2.INTER_AREA)
        if background.shape[2] == 3:
            background = np.dstack((background, np.full((1000, 1000), 255, dtype=np.uint8)))

        train_img_path = os.path.join(train_folder, train_images[large_idx])
        traindotted_img_path = os.path.join(traindotted_folder, traindotted_images[large_idx])
        train_img = cv2.imread(train_img_path, cv2.IMREAD_UNCHANGED)
        traindotted_img = cv2.imread(traindotted_img_path, cv2.IMREAD_UNCHANGED)

        if train_img is None or traindotted_img is None:
            print(f"Failed to load image: {train_img_path} or {traindotted_img_path}")
            continue

        # 調整大小並居中
        h, w = train_img.shape[:2]
        x = (1000 - w) // 2  # 更新為 1000
        y = (1000 - h) // 2
        x_end = min(x + w, 1000)
        y_end = min(y + h, 1000)
        w_adj = x_end - x
        h_adj = y_end - y

        train_img = train_img[:h_adj, :w_adj]
        traindotted_img = traindotted_img[:h_adj, :w_adj]

        train_mosaic = background.copy()
        traindotted_mosaic = background.copy()

        if train_img.shape[2] == 4:
            train_rgb = train_img[:, :, :3]
            train_alpha = train_img[:, :, 3] / 255.0
            traindotted_rgb = traindotted_img[:, :, :3]
            traindotted_alpha = traindotted_img[:, :, 3] / 255.0
        else:
            train_rgb = train_img
            train_alpha = np.ones((h_adj, w_adj), dtype=np.float32)
            traindotted_rgb = traindotted_img
            traindotted_alpha = np.ones((h_adj, w_adj), dtype=np.float32)

        background_region = train_mosaic[y:y+h_adj, x:x+w_adj, :3]
        background_region_dotted = traindotted_mosaic[y:y+h_adj, x:x+w_adj, :3]

        for c in range(3):
            train_mosaic[y:y+h_adj, x:x+w_adj, c] = (train_rgb[:, :, c] * train_alpha + background_region[:, :, c] * (1 - train_alpha)).astype(np.uint8)
            traindotted_mosaic[y:y+h_adj, x:x+w_adj, c] = (traindotted_rgb[:, :, c] * traindotted_alpha + background_region_dotted[:, :, c] * (1 - traindotted_alpha)).astype(np.uint8)
        train_mosaic[y:y+h_adj, x:x+w_adj, 3] = 255
        traindotted_mosaic[y:y+h_adj, x:x+w_adj, 3] = 255

        os.makedirs(os.path.join(output_root, "train_mosaic"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "traindotted_mosaic"), exist_ok=True)
        train_mosaic_path = os.path.join(output_root, "train_mosaic", f"mosaic_{mosaic_idx}_1000x1000.png")  # 更新為 1000x1000
        traindotted_mosaic_path = os.path.join(output_root, "traindotted_mosaic", f"mosaic_{mosaic_idx}_1000x1000.png")
        cv2.imwrite(train_mosaic_path, train_mosaic)
        cv2.imwrite(traindotted_mosaic_path, traindotted_mosaic)
        print(f"Saved Train mosaic to {train_mosaic_path}, size: 1000x1000")
        print(f"Saved TrainDotted mosaic to {traindotted_mosaic_path}, size: 1000x1000")
        mosaic_idx += 1

    # 處理剩餘的小圖片
    while remaining_small:
        current_boxes = []
        used_indices = []
        current_area = 0

        # 從面積最小開始加入，直到達到 80% 面積
        small_idx_pos = 0
        while small_idx_pos < len(remaining_small) and current_area < total_area_threshold:
            small_idx = remaining_small[small_idx_pos]
            small_box = boxes[small_idx]
            small_area = small_box[0] * small_box[1]
            if current_area + small_area <= total_area_threshold:
                current_boxes.append(small_box)
                used_indices.append(small_idx)
                current_area += small_area
                remaining_small.pop(small_idx_pos)
            else:
                small_idx_pos += 1

        if not current_boxes:
            break

        # 使用 phsppog 進行 packing
        height, rectangles = phsppog(1000, current_boxes, sorting='height')  # 更新為 1000
        print(f"Packing for mosaic {mosaic_idx}: height={height}, number of rectangles={len(rectangles)}, total area={current_area}")

        # 隨機選擇一張背景圖
        background_path = os.path.join(background_folder, random.choice(background_images))
        background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
        if background is None:
            print(f"Failed to load background image: {background_path}")
            background = np.zeros((1000, 1000, 4), dtype=np.uint8)  # 更新為 1000x1000
        elif background.shape[:2] != (1000, 1000):
            background = cv2.resize(background, (1000, 1000), interpolation=cv2.INTER_AREA)
        if background.shape[2] == 3:
            background = np.dstack((background, np.full((1000, 1000), 255, dtype=np.uint8)))

        # 初始化 1000x1000 背景圖
        train_mosaic = background.copy()
        traindotted_mosaic = background.copy()
        
        # 貼上子圖像
        for idx, rect in enumerate(rectangles):
            global_idx = used_indices[idx]
            if global_idx >= len(train_images):
                break
            x, y, w, h = rect
            # 讀取對應圖片
            train_img_path = os.path.join(train_folder, train_images[global_idx])
            traindotted_img_path = os.path.join(traindotted_folder, traindotted_images[global_idx])
            
            train_img = cv2.imread(train_img_path, cv2.IMREAD_UNCHANGED)
            traindotted_img = cv2.imread(traindotted_img_path, cv2.IMREAD_UNCHANGED)
            
            if train_img is None or traindotted_img is None:
                print(f"Failed to load image: {train_img_path} or {traindotted_img_path}")
                continue
            
            # 調整子圖像大小
            train_img = cv2.resize(train_img, (w, h), interpolation=cv2.INTER_AREA)
            traindotted_img = cv2.resize(traindotted_img, (w, h), interpolation=cv2.INTER_AREA)
            
            # 確保不超出 1000x1000
            x_end = min(x + w, 1000)  # 更新為 1000
            y_end = min(y + h, 1000)
            w_adj = x_end - x
            h_adj = y_end - y
            
            if w_adj <= 0 or h_adj <= 0:
                print(f"Warning: Invalid dimensions at x={x}, y={y}, w={w_adj}, h={h_adj}")
                continue
            
            # 提取子圖像的前景和透明通道
            train_img = train_img[:h_adj, :w_adj]
            traindotted_img = traindotted_img[:h_adj, :w_adj]
            
            if train_img.shape[2] == 4:
                train_rgb = train_img[:, :, :3]
                train_alpha = train_img[:, :, 3] / 255.0
                traindotted_rgb = traindotted_img[:, :, :3]
                traindotted_alpha = traindotted_img[:, :, 3] / 255.0
            else:
                train_rgb = train_img
                train_alpha = np.ones((h_adj, w_adj), dtype=np.float32)
                traindotted_rgb = traindotted_img
                traindotted_alpha = np.ones((h_adj, w_adj), dtype=np.float32)

            # 提取背景區域
            background_region = train_mosaic[y:y+h_adj, x:x+w_adj, :3]
            background_region_dotted = traindotted_mosaic[y:y+h_adj, x:x+w_adj, :3]

            # 根據透明通道進行混合
            for c in range(3):
                train_mosaic[y:y+h_adj, x:x+w_adj, c] = (train_rgb[:, :, c] * train_alpha + background_region[:, :, c] * (1 - train_alpha)).astype(np.uint8)
                traindotted_mosaic[y:y+h_adj, x:x+w_adj, c] = (traindotted_rgb[:, :, c] * traindotted_alpha + background_region_dotted[:, :, c] * (1 - traindotted_alpha)).astype(np.uint8)
            train_mosaic[y:y+h_adj, x:x+w_adj, 3] = 255
            traindotted_mosaic[y:y+h_adj, x:x+w_adj, 3] = 255
        
        # 儲存馬賽克圖像為 PNG
        os.makedirs(os.path.join(output_root, "train_mosaic"), exist_ok=True)
        os.makedirs(os.path.join(output_root, "traindotted_mosaic"), exist_ok=True)
        train_mosaic_path = os.path.join(output_root, "train_mosaic", f"mosaic_{mosaic_idx}_1000x1000.png")  # 更新為 1000x1000
        traindotted_mosaic_path = os.path.join(output_root, "traindotted_mosaic", f"mosaic_{mosaic_idx}_1000x1000.png")
        cv2.imwrite(train_mosaic_path, train_mosaic)
        cv2.imwrite(traindotted_mosaic_path, traindotted_mosaic)
        print(f"Saved Train mosaic to {train_mosaic_path}, size: 1000x1000")
        print(f"Saved TrainDotted mosaic to {traindotted_mosaic_path}, size: 1000x1000")
        
        mosaic_idx += 1

if __name__ == "__main__":
    train_folder = "OutputBboxes/split_bboxes/train"
    traindotted_folder = "OutputBboxes/split_bboxes/traindotted"
    background_folder = "OutputBboxes/backgrounds"
    output_root = "OutputMosaic/"
    create_mosaic(train_folder, traindotted_folder, background_folder, output_root)