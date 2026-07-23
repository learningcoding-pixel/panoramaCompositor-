import glob
import os
import cv2
import numpy as np
import sys
from paddleocr import PaddleOCR

ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        det_db_box_thresh=0.2,
        det_db_thresh=0.2,
        det_limit_side_len=4096,
        det_limit_type="max"
        )

def run(folder):
    print(sys.executable)
    print("Detecting text...")
    
    

    # -------------------------------------------------------
    # Parameters
    # -------------------------------------------------------
    TILE_SIZE = 4096 #2048
    OVERLAP = 1024 #512
    STEP = TILE_SIZE - OVERLAP

    SCALES = [1.0, 1.5, 2.0,3.0,4.0]

    os.makedirs("masks", exist_ok=True)

    # -------------------------------------------------------
    # Read images
    # -------------------------------------------------------
    image_paths = sorted(
        glob.glob(str(folder / "*.jpg")) 
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

                    result = ocr.ocr(scaled, cls=True)

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


    # Create transparent BGRA image
    transparent = np.zeros((H, W, 4), dtype=np.uint8)

    # Red (OpenCV uses BGR order)
    transparent[:, :, 2] = 255  # Red channel

    # Alpha channel = combined mask
    transparent[:, :, 3] = combined_mask

    # Save
    output_path = folder / "combined_text_mask.png"
    cv2.imwrite(str(output_path), transparent)
    return combined_mask

