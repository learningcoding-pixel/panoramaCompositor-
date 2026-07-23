from pathlib import Path

import hd
import inpaint
import medianStack
import td

ROOT = Path("panoramas")   # Folder containing scene1, scene2, ...

for folder in sorted(ROOT.iterdir()):

    if not folder.is_dir():
        continue

    if folder.name == "ALL":
        continue

    print(f"\n========== {folder.name} ==========")

    print("Running Human detection...")
    humanMasks = hd.run(folder)
    print("Human detection done.")

    print("Running median stacking...")
    missingHumans = medianStack.run(folder, humanMasks)
    print("Median stacking done.")

    print("Running text detection...")
    textMasks = td.run(folder)
    print("Text detection done.")

    print("Running inpainting...")
    inpaint.run(folder, missingHumans, textMasks)
    print("Inpainting done.")