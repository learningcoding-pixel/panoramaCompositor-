import glob
import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import torch
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest
print(sys.executable)

# ---------------------------------------
# Load images
# ---------------------------------------

image_paths = sorted(glob.glob("*.jpg") + glob.glob("*.png"))

images = [cv2.imread(f) for f in image_paths]

if len(images) == 0:
    raise Exception("No images found")

h, w = images[0].shape[:2]
images = [cv2.resize(img, (w, h)) for img in images]

# ---------------------------------------
# Load YOLO Segmentation
# ---------------------------------------

#model = YOLO("yolov8n-seg.pt")
model = YOLO("yolo11x-seg.pt")

# ---------------------------------------
# Load IOPaint (LaMa)
# ---------------------------------------

inpainter = ModelManager(
    name="lama",
    device=torch.device("cpu")   # use "cuda" if you have a supported NVIDIA GPU
)

# ---------------------------------------
# Human mask for every image
# ---------------------------------------

human_masks = []

for img_path, img in zip(image_paths, images):

    #result = model.predict(img, verbose=False)[0]
    #result = model.predict(img, conf=0.1, verbose=False)[0]
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

        for cls, poly in zip(result.boxes.cls.cpu().numpy(),
                             result.masks.xy):

            # COCO class 0 = person
            if int(cls) != 0:
                continue

            poly = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)

    human_masks.append(mask.astype(bool))

    # Get filename without extension
    name = os.path.splitext(os.path.basename(img_path))[0]

    # Save as imageName_mask.png
    cv2.imwrite(f"{name}_mask.png", mask)

# ---------------------------------------
# Median stack (ignore human pixels)
# ---------------------------------------

# Shape: (N, H, W, 3)
stack = np.stack(images, axis=0).astype(np.float32)

# Shape: (N, H, W)
masks = np.stack(human_masks, axis=0)

# Set human pixels to NaN
for i in range(len(images)):
    stack[i][masks[i]] = np.nan

# Compute median using only background pixels
median = np.nanmedian(stack, axis=0)
cv2.imwrite("median_before_lama.png", median)
# ---------------------------------------
# Fill pixels that were human in every image
# ---------------------------------------

# Pixels where the median could not be computed
missing = np.isnan(median[:, :, 0])



# Copy a valid background pixel from the original images
for img, human_mask in zip(images, human_masks):

    # Pixel is missing AND this image has background there
    valid = missing & (~human_mask)

    median[valid] = img[valid]

    missing[valid] = False

    if not missing.any():
        break



# ---------------------------------------
# Inpaint any pixels that were still missing
# ---------------------------------------


kernel = np.ones((11,11), np.uint8)

inpaint_mask = cv2.dilate(
    (missing.astype(np.uint8) * 255),
    kernel,
    iterations=2
)

# ---------------------------------------
# Convert median to uint8
# ---------------------------------------

#median = np.nan_to_num(median)          # Replace any remaining NaNs with 0
#median = np.clip(median, 0, 255)        # Ensure values are in valid range
#median = median.astype(np.uint8)

# Replace any remaining NaNs with temporary values
median = np.where(np.isnan(median), images[0], median)

# Convert to uint8
median = np.clip(median, 0, 255).astype(np.uint8)

# ---------------------------------------
# Inpaint using IOPaint (LaMa)
# ---------------------------------------

config = InpaintRequest(
    hd_strategy="Crop",
    hd_strategy_crop_margin=128,
    hd_strategy_crop_trigger_size=800,
    ldm_steps=20,
    ldm_sampler="plms"
)

final = inpainter(
    image=median,
    mask=inpaint_mask,
    config=config
)
####
# Convert RGB -> BGR before saving
final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
####
cv2.imwrite("final_result.png", final)
#difference = cv2.absdiff(median, final)
#cv2.imwrite("difference.png", difference)

print("Done!")
