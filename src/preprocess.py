import os
import warnings
import pandas as pd
import pydicom
from tqdm import tqdm

# -----------------------------
# Settings
# -----------------------------
RAW_DICOM_DIR = "data/raw"
LABELS_CSV = "data/labels.csv"
OUTPUT_CSV = "data/processed_labels.csv"

# -----------------------------
# Quiet the noisy pydicom warnings (the "Invalid VR UI" spam)
# -----------------------------
warnings.filterwarnings(
    "ignore",
    message=r"Invalid value for VR UI.*",
    category=UserWarning,
    module=r"pydicom.*",
)

def main():
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Missing {LABELS_CSV}. Run: python src/build_labels.py")

    if not os.path.isdir(RAW_DICOM_DIR):
        raise FileNotFoundError(f"Missing folder {RAW_DICOM_DIR}. Put .dcm files inside it.")

    print("[INFO] Loading labels.csv...")
    labels_df = pd.read_csv(LABELS_CSV)

    # Ensure correct columns exist
    if "ImageId" not in labels_df.columns or "Label" not in labels_df.columns:
        raise ValueError("labels.csv must have columns: ImageId, Label")

    # Map: SOPInstanceUID (ImageId) -> label (0/1)
    label_map = dict(zip(labels_df["ImageId"].astype(str), labels_df["Label"].astype(int)))
    print(f"[INFO] Loaded labels for {len(label_map)} unique images")

    # Collect all .dcm paths (fast)
    dcm_paths = []
    with os.scandir(RAW_DICOM_DIR) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".dcm"):
                dcm_paths.append(entry.path)

    print(f"[INFO] Found {len(dcm_paths)} DICOM files in {RAW_DICOM_DIR}")

    records = []
    read_fail = 0
    uid_missing = 0
    not_in_labels = 0
    matched = 0

    print("[INFO] Reading DICOM headers + matching SOPInstanceUID...")
    for dcm_path in tqdm(dcm_paths):
        try:
            # Read ONLY metadata (fastest); stop before pixels
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
        except Exception:
            read_fail += 1
            continue

        uid = getattr(dcm, "SOPInstanceUID", None)
        if not uid:
            uid_missing += 1
            continue

        uid = str(uid)

        if uid not in label_map:
            not_in_labels += 1
            continue

        # ✅ Match
        matched += 1
        records.append({
            "dicom_path": dcm_path,
            "uid": uid,
            "label": int(label_map[uid]),
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("\n================ SUMMARY ================")
    print(f"[✓] Saved {len(out_df)} matched samples to {OUTPUT_CSV}")
    print(f"Matched: {matched}")
    print(f"Read failures: {read_fail}")
    print(f"Missing SOPInstanceUID: {uid_missing}")
    print(f"UID not in labels.csv: {not_in_labels}")
    print("========================================\n")

    # Only print label counts if we actually have rows
    if len(out_df) > 0:
        print(out_df["label"].value_counts())

if __name__ == "__main__":
    main()
