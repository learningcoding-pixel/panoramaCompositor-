import glob
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

# Load YOLO Segmentation
model = YOLO("yolo11x-seg.pt")

def run(folder):
    print(sys.executable)
    print("Detecting people...")
    # ---------------------------------------
    # Load images
    # ---------------------------------------
    
    image_paths = sorted(glob.glob(str(folder / "*.jpg")) + glob.glob(str(folder / "*.png")))
    images = [cv2.imread(f) for f in image_paths]
    
    if len(images) == 0:
        raise Exception("No images found")
    
    h, w = images[0].shape[:2]
    images = [cv2.resize(img, (w, h)) for img in images]
    
    
    
    # ---------------------------------------
    # Human mask for every image
    # ---------------------------------------

    human_masks = []
    
    for img_path, img in zip(image_paths, images):
        result = model.predict(
        img,
        imgsz=2560,
        conf=0.01,
        iou=0.8,
        max_det=1000,
        agnostic_nms=True,
        verbose=False
        )[0]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if result.masks is not None:
            
            for cls, poly in zip(
                result.boxes.cls.cpu().numpy(),
                result.masks.xy
                ):
                # COCO class 0 = person
                if int(cls) != 0:
                    continue
                
                poly = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [poly], 255)
                
        human_masks.append(mask.astype(bool))
        # Create transparent BGRA image
        transparent = np.zeros((h, w, 4), dtype=np.uint8)

        # Red (OpenCV uses BGR order)
        transparent[:, :, 2] = 255   # Red channel

        # Alpha channel = mask
        transparent[:, :, 3] = mask

        name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = folder / f"{name}_mask.png"

        cv2.imwrite(str(output_path), transparent)

    print(len(images), "images processed.")
    print(len(human_masks), "human masks generated.")

    return human_masks
    





