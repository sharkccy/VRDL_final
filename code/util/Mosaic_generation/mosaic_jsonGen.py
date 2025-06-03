import cv2
import numpy as np
import skimage.feature
import os
import json
import glob
import math
import csv

# Parameters
bbox_size = 50  # Base bounding box size (pixels)
class_names = ['adult_male', 'subadult_male', 'adult_female', 'juvenile', 'pup']
category_to_expand = {
    "adult_male": 1.8,    # 成年雄性
    "subadult_male": 1.5, # 未成年雄性
    "adult_female": 1.5,  # 雌性
    "juvenile": 1.2,     # 少年
    "pup": 0.8          # 幼雛
}
mosaic_train_dir = "OutputMosaic/train_mosaic"
mosaic_dotted_dir = "OutputMosaic/traindotted_mosaic"
json_path = "mosaic_bboxes.json"
output_visual_dir_dotted = "OutputMosaic/traindotted_mosaic_with_bboxes"
output_csv_path = "train_mosaic.csv"

def extract_and_visualize_bboxes(
    filename,
    dotted_dir=mosaic_dotted_dir,
    train_dir=mosaic_train_dir,
    annotation_id_start=1,
    use_blob=True
):
    # Check image paths
    dotted_path = os.path.join(dotted_dir, filename)
    train_path = os.path.join(train_dir, filename)
    if not os.path.exists(dotted_path):
        raise FileNotFoundError(f"Cannot find TrainDotted mosaic image: {dotted_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Cannot find Train mosaic image: {train_path}")
    
    # Read images
    image_dotted = cv2.imread(dotted_path)
    image_train = cv2.imread(train_path)
    if image_dotted is None or image_train is None:
        raise ValueError(f"Failed to read images: {dotted_path} or {train_path}")
    
    # Check image sizes (ensure 1000x1000)
    if image_dotted.shape != (1000, 1000, 3) or image_train.shape != (1000, 1000, 3):
        print(f"Invalid size for {filename}: Dotted {image_dotted.shape}, Train {image_train.shape}, expected (1000, 1000, 3)")
        return None, None, None, None, None
    
    # Ensure both images are 3-channel (RGB)
    if image_dotted.shape[2] != 3:
        image_dotted = cv2.cvtColor(image_dotted, cv2.COLOR_GRAY2BGR if len(image_dotted.shape) == 2 else cv2.COLOR_BGRA2BGR)
    if image_train.shape[2] != 3:
        image_train = cv2.cvtColor(image_train, cv2.COLOR_GRAY2BGR if len(image_train.shape) == 2 else cv2.COLOR_BGRA2BGR)
    
    # Create copy for visualization (only for traindotted)
    image_dotted_with_bboxes = image_dotted.copy()
    
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

    # Define colors for each class (BGR format)
    bbox_colors = {
        0: (0, 0, 255),    # adult_male: Red
        1: (255, 0, 255),  # subadult_male: Magenta
        2: (42, 42, 165),  # adult_female: Brown
        3: (255, 0, 0),    # juvenile: Blue
        4: (0, 255, 0)     # pup: Green
    }

    # Detect dots
    bboxes = []
    annotations = []
    class_counts = {class_name: 0 for class_name in class_names}
    if use_blob:
        # Use blob detection with adjusted parameters
        blobs = skimage.feature.blob_log(image_gray, min_sigma=2, max_sigma=10, num_sigma=5, threshold=0.03)
        for blob in blobs:
            y, x, s = blob
            b, g, r = img_blur[int(y)][int(x)][:]  # Use blurred image colors
            print(f"Detected blob at ({x}, {y}): R={r}, G={g}, B={b}")  # Debug output

            # Adjusted color thresholds
            if 190 <= r <= 255 and b <= 50 and g <= 50:  # RED (adult_male), widened range
                class_id = 0
                expand_ratio = category_to_expand["adult_male"]
            elif 190 <= r <= 255 and 190 <= b <= 255 and g <= 50:  # MAGENTA (subadult_male), widened range
                class_id = 1
                expand_ratio = category_to_expand["subadult_male"]
            elif 40 <= r <= 120 and b <= 60 and g <= 100:  # BROWN (adult_female)
                class_id = 2
                expand_ratio = category_to_expand["adult_female"]
            elif 100 <= b <= 255 and r <= 100 and g <= 100:  # BLUE (juvenile)
                class_id = 3
                expand_ratio = category_to_expand["juvenile"]
            elif 150 <= g <= 255 and r <= 100 and b <= 100:  # GREEN (pup)
                class_id = 4
                expand_ratio = category_to_expand["pup"]
            else:
                print(f"Unmatched color at ({x}, {y}): R={r}, G={g}, B={b}")
                continue

            # Check if the blob is near the edge (likely to be incomplete)
            radius = s * np.sqrt(2)  # Approximate radius of the blob
            if (x - radius < 0 or x + radius > image_dotted.shape[1] or 
                y - radius < 0 or y + radius > image_dotted.shape[0]):
                print(f"Skipping potentially incomplete blob at ({x}, {y}): too close to edge, radius={radius}")
                continue

            # Calculate expected area of a circular blob
            expected_area = math.pi * (radius ** 2)

            # Extract the region around the blob to compute actual area
            x_min_area = max(0, int(x - radius))
            x_max_area = min(image_dotted.shape[1], int(x + radius))
            y_min_area = max(0, int(y - radius))
            y_max_area = min(image_dotted.shape[0], int(y + radius))
            blob_region = mask[y_min_area:y_max_area, x_min_area:x_max_area]
            actual_area = np.sum(blob_region > 0)  # Count non-zero pixels in the mask

            # Check if the blob is incomplete based on area ratio
            area_ratio = actual_area / expected_area
            if area_ratio < 0.7:  # If actual area is less than 70% of expected, assume incomplete
                print(f"Skipping incomplete blob at ({x}, {y}): area_ratio={area_ratio:.2f}, actual_area={actual_area}, expected_area={expected_area}")
                continue

            # Calculate bbox size based on category
            adjusted_bbox_size = int(bbox_size * expand_ratio)
            x_min = max(0, int(x - adjusted_bbox_size // 2))
            y_min = max(0, int(y - adjusted_bbox_size // 2))
            x_max = min(image_dotted.shape[1], int(x + adjusted_bbox_size // 2))
            y_max = min(image_dotted.shape[0], int(y + adjusted_bbox_size // 2))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            bboxes.append({
                "class_id": class_id,
                "bbox": bbox
            })
            # Update class count
            class_counts[class_names[class_id]] += 1
            # Add to COCO annotations
            annotations.append({
                "id": annotation_id_start + len(annotations),
                "image_id": int(filename.replace('_1000x1000.png', '').replace('mosaic_', '')),
                "category_id": class_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            # Draw bounding box on traindotted image
            top_left = (x_min, y_min)
            bottom_right = (x_max, y_max)
            color = bbox_colors[class_id]
            cv2.rectangle(image_dotted_with_bboxes, top_left, bottom_right, color, 2)

    return bboxes, annotations, {"id": int(filename.replace('_1000x1000.png', '').replace('mosaic_', '')), "file_name": filename, "width": image_dotted.shape[1], "height": image_dotted.shape[0]}, image_dotted_with_bboxes, class_counts

if __name__ == "__main__":
    # Directories and paths
    mosaic_train_dir = "OutputMosaic/train_mosaic"
    mosaic_dotted_dir = "OutputMosaic/traindotted_mosaic"
    output_visual_dir_dotted = "OutputMosaic/traindotted_mosaic_with_bboxes"
    json_path = "mosaic_bboxes.json"
    output_csv_path = "train_mosaic.csv"

    # Create output directory for visualized images (traindotted only)
    os.makedirs(output_visual_dir_dotted, exist_ok=True)

    # Initialize COCO data
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }

    # Initialize list to store class counts for CSV
    csv_data = []

    # Process all mosaic images
    annotation_id = 1
    try:
        for img_path in glob.glob(os.path.join(mosaic_dotted_dir, "mosaic_*.png")):
            filename = os.path.basename(img_path)
            try:
                bboxes, annotations, image_info, dotted_with_bboxes, class_counts = extract_and_visualize_bboxes(
                    filename,
                    dotted_dir=mosaic_dotted_dir,
                    train_dir=mosaic_train_dir,
                    annotation_id_start=annotation_id,
                    use_blob=True
                )
                if bboxes is None or annotations is None or image_info is None:
                    continue
                
                # Save traindotted image with bboxes
                dotted_output_path = os.path.join(output_visual_dir_dotted, filename)
                cv2.imwrite(dotted_output_path, dotted_with_bboxes)
                print(f"Saved TrainDotted mosaic with bboxes to {dotted_output_path}")

                # Add to COCO data
                coco_data["images"].append(image_info)
                coco_data["annotations"].extend(annotations)
                annotation_id += len(annotations)

                # Prepare row for CSV
                train_id = image_info["id"]
                csv_row = [train_id] + [class_counts[class_name] for class_name in class_names]
                csv_data.append(csv_row)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        # Save combined COCO JSON
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=4)
        print(f"Saved combined JSON to {json_path}")

        # Save class counts to CSV
        with open(output_csv_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["train_id", "adult_males", "subadult_males", "adult_females", "juveniles", "pups"])
            # Write data
            writer.writerows(csv_data)
        print(f"Saved class counts to {output_csv_path}")

    except Exception as e:
        print(f"Global error: {e}")