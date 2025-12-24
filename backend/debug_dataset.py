import os
import pandas as pd

RAW_DIR = "data/raw"
TRAIN_CSV = "data/train-rle.csv"

# Load CSV
df = pd.read_csv(TRAIN_CSV)

print("CSV ImageId sample:")
print(df["ImageId"].head(5).tolist())

# Find DICOM files
dicoms = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.endswith(".dcm"):
            dicoms.append(f)

print("\nNumber of DICOM files found:", len(dicoms))
print("Sample DICOM filenames:")
print(dicoms[:5])
