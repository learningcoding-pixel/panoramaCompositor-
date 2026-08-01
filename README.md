# Panorama Compositor

A Python-based panorama compositing pipeline that automatically removes moving people, transient objects, and unwanted text from panoramic image sequences. The pipeline combines deep learning models with classical image processing to reconstruct a clean background panorama.

---

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Ultralytics YOLO
- PaddleOCR
- PyTorch
- LaMa (IOPaint)

Install the required packages:

```bash
pip install ultralytics paddleocr opencv-python numpy torch
```

Install LaMa/IOPaint according to the official installation instructions.

---

## Features

- **Human Detection** using the YOLO segmentation model.
- **Median Stacking** to reconstruct static backgrounds while removing moving people and objects.
- **Text Detection** using PaddleOCR.
- **LaMa Inpainting** to fill regions where no valid background pixels exist.
- Processes multiple panorama folders automatically.
- Generates a final cleaned panorama ready for further visualization or analysis.

---

## Pipeline

```text
Panorama Images
        │
        ▼
YOLO Human Detection
        │
        ▼
Generate Human Masks
        │
        ▼
Median Stack Background
(Removes moving humans/objects)
        │
        ▼
Detect Missing Pixels
        │
        ▼
PaddleOCR Text Detection
        │
        ▼
Combine Missing Region Mask
        +
      Text Mask
        │
        ▼
LaMa Inpainting
        │
        ▼
Final Panorama
```

---

## Methodology

### 1. Human Detection

Each panorama image is processed using a YOLO segmentation model.

The model identifies all human instances and produces binary segmentation masks. These masks are used to exclude foreground pixels from the background reconstruction process.

**Output**

- Human segmentation masks
- Original images

---

### 2. Median Stack Background Reconstruction

The human masks are applied to each image.

Pixels belonging to detected humans are ignored by replacing them with `NaN` values. The background is reconstructed using the median value of all valid pixels across the image sequence.

```python
stack[i][human_mask] = np.nan
median = np.nanmedian(stack, axis=0)
```

If some pixels remain missing because every image contains a person at that location, valid background pixels from other images are copied whenever possible.

Remaining holes are recorded as an inpainting mask.

---

### 3. Text Detection

PaddleOCR scans the reconstructed panorama for text.

Detected text regions are converted into a binary mask which identifies:

- Street signs
- Building names
- Advertisements
- Other visible text

The mask is dilated slightly to ensure complete removal.

---

### 4. LaMa Inpainting

The missing-region mask from median stacking is combined with the PaddleOCR text mask.

LaMa then reconstructs these regions using surrounding image context.

This removes:

- Remaining people
- Moving objects
- Missing background areas
- Detected text

while preserving surrounding textures.

---

## Technologies

| Component | Model / Library |
|-----------|-----------------|
| Human Detection | YOLO Segmentation |
| Image Processing | OpenCV |
| Numerical Computation | NumPy |
| Text Detection | PaddleOCR |
| Inpainting | LaMa |
| Programming Language | Python |

---

## Project Structure

```text
project/
│
├── panoramas/
│   ├── scene1/
│   ├── scene2/
│   └── scene3/
│
├── hd.py
├── inpaint.py
├── main.py
├── medianStack.py
├── td.py
```

---

## Processing Workflow

1. Load panorama image sequence.
2. Detect humans using YOLO segmentation.
3. Generate human masks.
4. Perform median stacking while ignoring masked pixels.
5. Recover remaining valid background pixels.
6. Detect text using PaddleOCR.
7. Merge text mask with missing-region mask.
8. Perform LaMa inpainting.
9. Save the completed panorama.

---

## Output

The pipeline produces:

- Human segmentation masks
- Median stacked panorama
- Text mask
- Inpainting mask
- Final cleaned panorama



## Applications

This panorama compositor can be used for:

- Virtual tours
- Background reconstruction from panoramic image sequences