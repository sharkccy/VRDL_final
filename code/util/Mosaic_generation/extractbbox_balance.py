import os
import cv2
from collections import defaultdict
from pycocotools.coco import COCO

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_and_crop_bboxes(
    json_path,
    train_root,
    traindotted_root,
    output_root,
    max_per_class=8000,
    min_per_class=4000,
    expansion_ratios=None
):
    if expansion_ratios is None:
        expansion_ratios = {
            "adult_male": 1.8,
            "subadult_male": 1.8,
            "female": 1.8,
            "juvenile": 1.8,
            "pup": 1.8
        }

    # 定義類別名稱（與 YOLO 分類一致）
    class_names = ["adult_male", "subadult_male", "female", "juvenile", "pup"]

    # 原始輸出路徑
    coco = COCO(json_path)
    output_train = os.path.join(output_root, "train")
    output_traindotted = os.path.join(output_root, "traindotted")
    
    # 為每個類別創建子資料夾
    for class_name in class_names:
        ensure_dir(os.path.join(output_train, class_name))
        ensure_dir(os.path.join(output_traindotted, class_name))

    # 複製輸出路徑
    output_copies_root = os.path.join(output_root + "_copies")
    output_copies_train = os.path.join(output_copies_root, "train")
    output_copies_traindotted = os.path.join(output_copies_root, "traindotted")
    
    for class_name in class_names:
        ensure_dir(os.path.join(output_copies_train, class_name))
        ensure_dir(os.path.join(output_copies_traindotted, class_name))

    output_counts = defaultdict(int)

    # 定義需要複製的類別和複製次數
    copy_counts = {
        "adult_male": 2,
        "subadult_male": 2
    }

    for img in coco.loadImgs(coco.getImgIds()):
        img_id = img["id"]
        file_name = img["file_name"]

        img_train = cv2.imread(os.path.join(train_root, file_name))
        img_dotted = cv2.imread(os.path.join(traindotted_root, file_name))

        if img_train is None or img_dotted is None:
            print(f"圖片 {file_name} 無法載入，跳過")
            continue

        img_h, img_w = img_train.shape[:2]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            x, y, w, h = ann["bbox"]
            category = coco.loadCats([ann["category_id"]])[0]["name"]
            if category not in expansion_ratios:
                print(f"類別 {category} 不在 expansion_ratios 中，跳過")
                continue

            if output_counts[category] >= max_per_class:
                continue

            ratio = expansion_ratios[category]
            x_center = x + w / 2
            y_center = y + h / 2
            new_w = w * ratio
            new_h = h * ratio
            new_x = max(0, int(x_center - new_w / 2))
            new_y = max(0, int(y_center - new_h / 2))
            new_x2 = min(img_w, int(new_x + new_w))
            new_y2 = min(img_h, int(new_y + new_h))

            crop_w = new_x2 - new_x
            crop_h = new_y2 - new_y
            if crop_w <= 0 or crop_h <= 0:
                print(f"圖片 {file_name} 的裁剪區域無效，跳過")
                continue

            crop_train = img_train[new_y:new_y2, new_x:new_x2]
            crop_dotted = img_dotted[new_y:new_y2, new_x:new_x2]
            out_name = f"{img_id}_{new_x}_{new_y}_{crop_h}_{crop_w}.png"

            # 儲存原始裁剪圖片到類別子資料夾
            cv2.imwrite(os.path.join(output_train, category, out_name), crop_train)
            cv2.imwrite(os.path.join(output_traindotted, category, out_name), crop_dotted)
            output_counts[category] += 1

            # 檢查是否需要複製
            if category in copy_counts:
                num_copies = copy_counts[category]
                for i in range(num_copies):
                    copy_out_name = f"{img_id}_{new_x}_{new_y}_{crop_h}_{crop_w}_copy{i+1}.png"
                    cv2.imwrite(os.path.join(output_copies_train, category, copy_out_name), crop_train)
                    cv2.imwrite(os.path.join(output_copies_traindotted, category, copy_out_name), crop_dotted)
                    output_counts[category] += 1

        # 檢查是否所有類別都達到最小數量
        if all(count >= min_per_class for count in output_counts.values()):
            break

    print("各類別裁切圖片數量（包含複製）：")
    for category, count in output_counts.items():
        print(f"{category}: {count}")

if __name__ == "__main__":
    extract_and_crop_bboxes(
        json_path="all_bboxes.json",
        train_root="Train/",
        traindotted_root="TrainDotted/",
        output_root="Outputbboxes/cropped_bboxes"
    )