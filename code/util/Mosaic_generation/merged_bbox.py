import os
import cv2
import numpy as np
from collections import defaultdict

def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    return {
        "image_id": parts[0],
        "xmin": int(parts[1]),
        "ymin": int(parts[2]),
        "h": int(parts[3]),
        "w": int(parts[4]),
        "filename": filename
    }

def check_overlap(b1, b2):
    x1, y1, x2, y2 = b1
    x1_, y1_, x2_, y2_ = b2
    return not (x2 <= x1_ or x2_ <= x1 or y2 <= y1_ or y2_ <= y1)

def merge_two_boxes(b1, b2):
    x1 = min(b1[0], b2[0])
    y1 = min(b1[1], b2[1])
    x2 = max(b1[2], b2[2])
    y2 = max(b1[3], b2[3])
    return [x1, y1, x2, y2]

def merge_and_save(group, train_dir, dotted_dir, save_dir_train, save_dir_dotted, image_id):
    used = [False] * len(group)
    merged_count = 0

    for i in range(len(group)):
        if used[i]:
            continue
        b1 = group[i]
        x1, y1, h1, w1 = b1["xmin"], b1["ymin"], b1["h"], b1["w"]
        x2, y2 = x1 + w1, y1 + h1
        merged_box = [x1, y1, x2, y2]
        bbox_group = [b1]
        used[i] = True

        for j in range(i + 1, len(group)):
            if used[j]:
                continue
            b2 = group[j]
            xx1, yy1, hh2, ww2 = b2["xmin"], b2["ymin"], b2["h"], b2["w"]
            xx2, yy2 = xx1 + ww2, yy1 + hh2

            if check_overlap(merged_box, [xx1, yy1, xx2, yy2]):
                bbox_group.append(b2)
                merged_box = merge_two_boxes(merged_box, [xx1, yy1, xx2, yy2])
                used[j] = True

        x_start, y_start, x_end, y_end = merged_box
        merge_w = x_end - x_start
        merge_h = y_end - y_start

        canvas_train = np.zeros((merge_h, merge_w, 4), dtype=np.uint8)
        canvas_dotted = np.zeros((merge_h, merge_w, 4), dtype=np.uint8)

        for b in bbox_group:
            fname = b["filename"]
            orig_x, orig_y = b["xmin"], b["ymin"]
            subimg_train = cv2.imread(os.path.join(train_dir, fname), cv2.IMREAD_UNCHANGED)
            subimg_dotted = cv2.imread(os.path.join(dotted_dir, fname), cv2.IMREAD_UNCHANGED)

            for canvas, subimg in [(canvas_train, subimg_train), (canvas_dotted, subimg_dotted)]:
                paste_x = orig_x - x_start
                paste_y = orig_y - y_start
                h, w = subimg.shape[:2]

                if subimg.shape[2] == 3:
                    alpha_channel = np.ones((h, w, 1), dtype=np.uint8) * 255
                    subimg = np.concatenate((subimg, alpha_channel), axis=2)

                canvas[paste_y:paste_y+h, paste_x:paste_x+w] = subimg

        save_name = f"{image_id}_{x_start}_{y_start}_{merge_h}_{merge_w}.png"
        cv2.imwrite(os.path.join(save_dir_train, save_name), canvas_train)
        cv2.imwrite(os.path.join(save_dir_dotted, save_name), canvas_dotted)
        merged_count += 1

    return merged_count

def merge_bboxes_both_views(
    train_dir="OutputBboxes/cropped_bboxes/train",
    dotted_dir="OutputBboxes/cropped_bboxes/traindotted",
    output_dir_train="OutputBboxes/merged_bboxes/train",
    output_dir_dotted="OutputBboxes/merged_bboxes/traindotted"
):
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_dotted, exist_ok=True)

    groups = defaultdict(list)
    for fname in os.listdir(train_dir):
        if fname.endswith(".jpg"):
            info = parse_filename(fname)
            groups[info["image_id"]].append(info)

    total_merged = 0
    for image_id, group in groups.items():
        total_merged += merge_and_save(
            group, train_dir, dotted_dir, output_dir_train, output_dir_dotted, image_id
        )

    print(f"完成合併，train 與 traindotted 各產生 {total_merged} 張 merged bbox 圖片。")

if __name__ == "__main__":
    merge_bboxes_both_views()
