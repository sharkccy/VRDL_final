import cv2
import numpy as np
import skimage.feature
import os
import pandas as pd
import json
import glob

# Parameters
bbox_size = 50  # Bounding box size (pixels)
class_names = ['adult_male', 'subadult_male', 'pup', 'juvenile', 'female']
train_csv_path = "train.csv"  # Path to train.csv
count_diff_threshold = 0.15  # Difference threshold as a fraction of train.csv count (10%)
min_diff_threshold = 20  # Minimum absolute difference threshold

def extract_and_visualize_bboxes(
    filename,
    dotted_dir="TrainDotted",
    train_dir="Train",
    image_id=None,  # 基於 filename 計算
    annotation_id_start=1,
    use_blob=True
):
    # Check image paths
    dotted_path = os.path.join(dotted_dir, filename)
    train_path = os.path.join(train_dir, filename)
    if not os.path.exists(dotted_path):
        raise FileNotFoundError(f"Cannot find TrainDotted image: {dotted_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Cannot find Train image: {train_path}")
    
    # Read images
    image_dotted = cv2.imread(dotted_path)
    image_train = cv2.imread(train_path)
    if image_dotted is None or image_train is None:
        raise ValueError(f"Failed to read images: {dotted_path} or {train_path}")
    
    # Check image sizes
    if image_dotted.shape != image_train.shape:
        print(f"Size mismatch for {filename}: Dotted {image_dotted.shape}, Train {image_train.shape}")
        return None, None, None  # Skip this image
    
    # Ensure both images are 3-channel (RGB)
    if image_dotted.shape[2] != 3:
        image_dotted = cv2.cvtColor(image_dotted, cv2.COLOR_GRAY2BGR if len(image_dotted.shape) == 2 else cv2.COLOR_BGRA2BGR)
    if image_train.shape[2] != 3:
        image_train = cv2.cvtColor(image_train, cv2.COLOR_GRAY2BGR if len(image_train.shape) == 2 else cv2.COLOR_BGRA2BGR)
    
    # Apply Gaussian blur to TrainDotted (for color classification)
    img_blur = cv2.GaussianBlur(image_dotted, (5, 5), 0)

    # Compute image difference to highlight dots
    image_diff = cv2.absdiff(image_dotted, image_train)
    
    # Generate mask (based on TrainDotted grayscale)
    mask = cv2.cvtColor(image_dotted, cv2.COLOR_BGR2GRAY)
    mask[mask < 50] = 0
    mask[mask > 0] = 255
    
    # Apply mask to difference image
    image_masked = cv2.bitwise_or(image_diff, image_diff, mask=mask)
    
    # Use max channel as blob_log input
    image_gray = np.max(image_masked, axis=2)

    # Detect dots
    bboxes = []
    annotations = []
    if use_blob:
        # Use blob detection
        blobs = skimage.feature.blob_log(image_gray, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)
        for blob in blobs:
            y, x, s = blob
            b, g, R = img_blur[int(y)][int(x)][:]  # Use blurred image colors
            # Color thresholds
            if R > 225 and b < 25 and g < 25:  # RED
                class_id = 0  # adult_male
            elif R > 225 and b > 225 and g < 25:  # MAGENTA
                class_id = 1  # subadult_male
            elif R < 75 and b < 50 and 150 < g < 200:  # GREEN
                class_id = 4  # pup
            elif R < 75 and 150 < b < 200 and g < 75:  # BLUE
                class_id = 3  # juvenile
            elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
                class_id = 2  # female
            else:
                continue
            # Generate bounding box
            x_min = max(0, int(x - bbox_size // 2))
            y_min = max(0, int(y - bbox_size // 2))
            x_max = min(image_dotted.shape[1], int(x + bbox_size // 2))
            y_max = min(image_dotted.shape[0], int(y + bbox_size // 2))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            bboxes.append({
                "class_id": class_id,
                "bbox": bbox
            })
            # Add to COCO annotations
            annotations.append({
                "id": annotation_id_start + len(annotations),
                "image_id": int(filename.replace('.jpg', '')),  # 基於 filename 提取 image_id
                "category_id": class_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })

    # Validate bounding box count with train.csv
    if os.path.exists(train_csv_path):
        try:
            train_csv = pd.read_csv(train_csv_path)
            # Extract numeric part from filename (e.g., '0.jpg' -> '0')
            train_id = filename.replace('.jpg', '')
            # Ensure train_id is string for consistent comparison
            train_csv['train_id'] = train_csv['train_id'].astype(str)
            counts = train_csv[train_csv['train_id'] == train_id]
            if not counts.empty:
                total_count = counts[['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']].sum(axis=1).sum()
                detected_count = len(bboxes)
                count_diff = abs(total_count - detected_count)
                threshold = min(count_diff_threshold * total_count, min_diff_threshold)
                if count_diff > threshold:
                    print(f"Invalid count for {filename}: Detected {detected_count}, train.csv {total_count}, difference {count_diff} exceeds threshold {threshold}")
                    return None, None, None  # Skip this image
                print(f"Valid count for {filename}: Detected {detected_count}, train.csv {total_count}")
            else:
                print(f"No count found in train.csv for {train_id}")
                return None, None, None  # Skip this image
        except Exception as e:
            print(f"Error reading train.csv for {filename}: {e}")
            return None, None, None  # Skip this image
    else:
        print(f"Cannot find train.csv: {train_csv_path}")
        return None, None, None  # Skip this image

    return bboxes, annotations, {"id": int(filename.replace('.jpg', '')), "file_name": filename, "width": image_dotted.shape[1], "height": image_dotted.shape[0]}

if __name__ == "__main__":
    # Directories and paths
    dotted_dir = "TrainDotted"
    train_dir = "Train"
    json_path = "all_bboxes.json"

    # Initialize COCO data
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }

    # Process all images
    annotation_id = 1
    try:
        for img_path in glob.glob(os.path.join(dotted_dir, "*.jpg")):
            filename = os.path.basename(img_path)
            try:
                bboxes, annotations, image_info = extract_and_visualize_bboxes(
                    filename,
                    dotted_dir=dotted_dir,
                    train_dir=train_dir,
                    annotation_id_start=annotation_id,
                    use_blob=True
                )
                if bboxes is None or annotations is None or image_info is None:
                    continue  # Skip invalid images
                coco_data["images"].append(image_info)
                coco_data["annotations"].extend(annotations)
                annotation_id += len(annotations)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        # Save combined COCO JSON
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=4)
        print(f"Saved combined JSON to {json_path}")

    except Exception as e:
        print(f"Global error: {e}")