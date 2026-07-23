import cv2
import numpy as np
import torch
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest

# ---------------------------------------
# Load IOPaint (LaMa)
# ---------------------------------------
    
inpainter = ModelManager(
    name="lama",
    device=torch.device("cpu")   # use "cuda" if you have a supported NVIDIA GPU
)

def run(folder,missingHumans, text_masks):
    
    globalMask = cv2.bitwise_or(missingHumans, text_masks)

    # ---------------------------------------
    # Load images
    # ---------------------------------------

    image = cv2.imread(str(folder / "median_before_lama.png"))

    

    kernel = np.ones((11,11), np.uint8)
    
    inpaint_mask = cv2.dilate(
        (globalMask.astype(np.uint8)),
        kernel,
        iterations=2
    )

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
        image=image,
        mask=inpaint_mask,
        config=config
    )

    # Convert RGB -> BGR before saving
    final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)

    # Save in the current folder
    folder_output = folder / f"{folder.name}.jpg"

    # Create panoramas/ALL if it doesn't exist
    all_folder = folder.parent / "ALL"
    all_folder.mkdir(exist_ok=True)

    # Save in panoramas/ALL
    all_output = all_folder / f"{folder.name}.jpg"

    # Write both images
    cv2.imwrite(
        str(folder_output),
        final,
        [cv2.IMWRITE_JPEG_QUALITY, 100]
    )

    cv2.imwrite(
        str(all_output),
        final,
        [cv2.IMWRITE_JPEG_QUALITY, 100]
   )

    print("Done!")





if __name__ == "__main__":
    run([])