import os
import cv2
import random
from pycocotools.coco import COCO

def boxes_outside_patch(patch, bboxes):
    """檢查所有 bbox 是否完全位於 patch 區域外部"""
    px, py, pw, ph = patch
    for bx, by, bw, bh in bboxes:
        # bbox 完全外部的條件：bbox 左邊或右邊超出 patch，或 bbox 上邊或下邊超出 patch
        if not (px + pw <= bx or bx + bw <= px or py + ph <= by or by + bh <= py):
            return False
    return True

def get_random_background_patch(image, bboxes, patch_size=1000, max_trials=100):
    h, w = image.shape[:2]
    for _ in range(max_trials):
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        candidate_patch = [x, y, patch_size, patch_size]
        if boxes_outside_patch(candidate_patch, bboxes):
            return image[y:y+patch_size, x:x+patch_size]
    return None

def extract_clean_backgrounds(json_path, image_root, output_dir, target_patches=100):
    coco = COCO(json_path)
    os.makedirs(output_dir, exist_ok=True)
    background_id = 0

    for img_info in coco.loadImgs(coco.getImgIds()):
        if background_id >= target_patches:
            break

        img_path = os.path.join(image_root, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            continue

        ann_ids = coco.getAnnIds(imgIds=img_info["id"])
        anns = coco.loadAnns(ann_ids)
        bboxes = [ann["bbox"] for ann in anns]  # [x, y, w, h]

        for _ in range(3):  # 每張圖最多嘗試3次找背景
            if background_id >= target_patches:
                break
            patch = get_random_background_patch(image, bboxes)
            if patch is not None:
                out_path = os.path.join(output_dir, f"background_{background_id}.png")
                cv2.imwrite(out_path, patch)
                background_id += 1

    print(f"總共成功產生了 {background_id} 張背景圖（目標：{target_patches}）")

if __name__ == "__main__":
    extract_clean_backgrounds(
        json_path="all_bboxes.json",
        image_root="Train/",
        output_dir="OutputBboxes/backgrounds/",
        target_patches=100
    )