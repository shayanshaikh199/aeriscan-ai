"""
preprocess.py

Aeriscan AI
-------------
Converts raw DICOM chest X-rays into PNG images
and generates binary labels for pneumothorax detection.

Label logic:
- EncodedPixels == "-1" → No pneumothorax (0)
- Otherwise → Pneumothorax present (1)
"""

import os
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

RAW_DIR = "data/raw"
IMAGE_DIR = "data/images"
LABELS_PATH = "data/labels.csv"
TRAIN_CSV = "data/train-rle.csv"

os.makedirs(IMAGE_DIR, exist_ok=True)


def dicom_to_png(dcm_path: str, png_path: str):
    """Convert a DICOM image to normalized PNG"""
    dcm = pydicom.dcmread(dcm_path)
    image = dcm.pixel_array.astype(np.float32)

    # Normalize to 0–255
    image -= image.min()
    image /= image.max()
    image *= 255.0

    image = image.astype(np.uint8)
    Image.fromarray(image).convert("L").save(png_path)


def main():
    df = pd.read_csv(TRAIN_CSV)

    # Recursively index all DICOM files
    dicom_index = {}

    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            if file.endswith(".dcm"):
                key = os.path.splitext(file)[0]
                dicom_index[key] = os.path.join(root, file)

    print(f"[INFO] Found {len(dicom_index)} DICOM files")

    # Create binary labels
    df["label"] = df["EncodedPixels"].apply(
        lambda x: 0 if x == "-1" else 1
    )

    samples = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row["ImageId"]
        label = row["label"]

        if image_id not in dicom_index:
            continue  # skip safely

        dcm_file = dicom_index[image_id]
        png_file = os.path.join(IMAGE_DIR, f"{image_id}.png")

        if not os.path.exists(png_file):
            dicom_to_png(dcm_file, png_file)

        samples.append({
            "image": f"{image_id}.png",
            "label": label
        })

    pd.DataFrame(samples).to_csv(LABELS_PATH, index=False)
    print(f"[✓] Saved {len(samples)} samples to {LABELS_PATH}")


if __name__ == "__main__":
    main()
