import glob
import os



#import sys
import torch
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest

import cv2
import numpy as np
from paddleocr import PaddleOCR


# -------------------------------------------------------
# Initialize OCR
# -------------------------------------------------------
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    det_limit_side_len=2048,
    det_limit_type="max"
)

# -------------------------------------------------------
# Parameters
# -------------------------------------------------------
TILE_SIZE = 2048
OVERLAP = 512
STEP = TILE_SIZE - OVERLAP

SCALES = [1.0, 1.5, 2.0]

os.makedirs("masks", exist_ok=True)

# -------------------------------------------------------
# Read images
# -------------------------------------------------------
image_paths = sorted(
    glob.glob("*.png") +
    glob.glob("*.jpg") +
    glob.glob("*.jpeg")
)

if len(image_paths) == 0:
    raise Exception("No images found.")

# Read first image to determine image size
first_image = cv2.imread(image_paths[0])

if first_image is None:
    raise Exception("Could not read first image.")

H, W = first_image.shape[:2]

# Combined mask for all images
combined_mask = np.zeros((H, W), dtype=np.uint8)

# -------------------------------------------------------
# Process images
# -------------------------------------------------------
for image_path in image_paths:

    print(f"\nProcessing {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        continue

    # Ensure all images are the same size
    if image.shape[:2] != (H, W):
        raise ValueError(f"{image_path} has a different resolution.")

    # Mask for this image only
    global_mask = np.zeros((H, W), dtype=np.uint8)

    total_detections = 0

    # ---------------------------------------------------
    # Tile loop
    # ---------------------------------------------------
    for y in range(0, H, STEP):

        for x in range(0, W, STEP):

            tile = image[
                y:min(y + TILE_SIZE, H),
                x:min(x + TILE_SIZE, W)
            ]

            if tile.size == 0:
                continue

            for scale in SCALES:

                if scale == 1.0:
                    scaled = tile
                else:
                    scaled = cv2.resize(
                        tile,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC
                    )

                result = ocr.ocr(scaled, cls=False)

                if result[0] is None:
                    continue

                total_detections += len(result[0])

                for line in result[0]:

                    pts = np.array(line[0], dtype=np.float32)

                    pts /= scale
                    pts[:, 0] += x
                    pts[:, 1] += y

                    pts = pts.astype(np.int32)

                    cv2.fillPoly(global_mask, [pts], 255)

    # Expand mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    global_mask = cv2.dilate(
        global_mask,
        kernel,
        iterations=2
    )

    # Accumulate into one combined mask
    combined_mask = cv2.bitwise_or(combined_mask, global_mask)

    filename = os.path.splitext(os.path.basename(image_path))[0]

    output_path = os.path.join(
        "masks",
        filename + "_mask.png"
    )

    cv2.imwrite(output_path, global_mask)

    print(f"Detected {total_detections} text regions")
    print(f"Saved: {output_path}")

# Save combined mask
cv2.imwrite("combined_mask.png", combined_mask)

# ---------------------------------------
# Inpaint using IOPaint (LaMa)
# ---------------------------------------

inpainter = ModelManager(
    name="lama",
    device=torch.device("cpu")   # use "cuda" if you have a supported NVIDIA GPU
)

config = InpaintRequest(
    hd_strategy="Crop",
    hd_strategy_crop_margin=128,
    hd_strategy_crop_trigger_size=800,
    ldm_steps=20,
    ldm_sampler="plms"
)

image = cv2.imread(image_paths[0])


final = inpainter(
    image=image,
    mask=combined_mask,
    config=config
)
####
# Convert RGB -> BGR before saving
final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
####
cv2.imwrite("final_result.png", final)



print("\nFinished.")