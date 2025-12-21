import pandas as pd

INPUT_CSV = "data/stage_2_train.csv"
OUTPUT_CSV = "data/labels.csv"

def main():
    print("[INFO] Loading stage_2_train.csv...")
    df = pd.read_csv(INPUT_CSV)

    # Pneumothorax label logic:
    # EncodedPixels == -1  → No pneumothorax (0)
    # Otherwise            → Pneumothorax (1)
    df["Label"] = df["EncodedPixels"].apply(lambda x: 0 if x == "-1" else 1)

    labels_df = df[["ImageId", "Label"]].drop_duplicates()

    labels_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[✓] labels.csv created with {len(labels_df)} samples")
    print(labels_df["Label"].value_counts())

if __name__ == "__main__":
    main()
