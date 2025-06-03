from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    model = YOLO("yolo11s-cls.yaml")
    results = model.train(
        data="train",
        epochs=150,
        patience = 30,
        lr0 = 0.001, 
        imgsz=100, 
        mosaic=0,
        warmup_epochs=30,
        close_mosaic=100,
        flipud=0.5,       # vertical flip augmentation
        fliplr=0.5,       # horizontal flip augmentation
        hsv_h=0.0,      # hue augmentation
        hsv_s=0.0,        # saturation augmentation
        hsv_v=0.0,        # brightness augmentation
        workers=8,
        batch=64, 
        erasing=0.0,
        conf=0.25,      # confidence threshold
        device="0"        # 0 for first GPU
    )