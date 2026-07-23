import glob
import cv2
import numpy as np

def run(folder, human_masks):
    print("median stacking...")
    
    # ---------------------------------------
    # Load images
    # ---------------------------------------

    image_paths = sorted(glob.glob(str(folder / "*.jpg")))

    images = [cv2.imread(f) for f in image_paths]
    
    if len(images) == 0:
        raise Exception("No images found")

    h, w = images[0].shape[:2]
    images = [cv2.resize(img, (w, h)) for img in images]
    
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

    
    cv2.imwrite(str(folder / "median_before_lama.png"), median)

    # Pixels where the median could not be computed
    missing = np.isnan(median[:, :, 0])


    # Convert bool -> uint8 mask
    missing = (missing.astype(np.uint8) * 255)

    return missing








