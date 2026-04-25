import os
import glob
import random
import pandas as pd


random_seed = 42
data_root = "NIH"
data_entry_csv = os.path.join(data_root, "Data_Entry_2017.csv")
train_csv = "train.csv"
val_csv = "val.csv"
val_ratio = 0.2


def build_image_lookup():
    # Find all PNG files only inside the NIH folder
    image_paths = glob.glob(os.path.join(data_root, "**", "*.png"), recursive=True)

    image_lookup = {}
    index = 0
    while index < len(image_paths):
        path = image_paths[index]
        filename = os.path.basename(path)
        image_lookup[filename] = path
        index += 1

    print("Found", len(image_lookup), "image files")
    return image_lookup


def make_binary_splits():
    if not os.path.exists(data_entry_csv):
        raise FileNotFoundError(f"Could not find {data_entry_csv}")

    dataframe = pd.read_csv(data_entry_csv)
    image_lookup = build_image_lookup()

    rows = []
    missing_images = 0

    for _, row in dataframe.iterrows():
        image_id = row["Image Index"]
        finding_labels = row["Finding Labels"]
        patient_id = row["Patient ID"]

        if image_id not in image_lookup:
            missing_images += 1
            continue

        # No Finding = normal = 0
        # anything else = abnormal = 1
        if finding_labels == "No Finding":
            label = 0
        else:
            label = 1

        rows.append({
            "image_path": image_lookup[image_id],
            "label": label,
            "patient_id": patient_id,
        })

    final_dataframe = pd.DataFrame(rows)

    print("Rows matched to images:", len(final_dataframe))
    print("Rows missing image files:", missing_images)

    # Split by patient to avoid leakage
    unique_patients = list(final_dataframe["patient_id"].unique())
    random.Random(random_seed).shuffle(unique_patients)

    split_index = int((1.0 - val_ratio) * len(unique_patients))
    train_patients = set(unique_patients[:split_index])
    val_patients = set(unique_patients[split_index:])

    train_dataframe = final_dataframe[
        final_dataframe["patient_id"].isin(train_patients)
    ].copy()

    val_dataframe = final_dataframe[
        final_dataframe["patient_id"].isin(val_patients)
    ].copy()

    train_dataframe = train_dataframe[["image_path", "label"]]
    val_dataframe = val_dataframe[["image_path", "label"]]

    train_dataframe.to_csv(train_csv, index=False)
    val_dataframe.to_csv(val_csv, index=False)

    print("Saved", train_csv, "with", len(train_dataframe), "rows")
    print("Saved", val_csv, "with", len(val_dataframe), "rows")

    train_normal = int((train_dataframe["label"] == 0).sum())
    train_abnormal = int((train_dataframe["label"] == 1).sum())
    val_normal = int((val_dataframe["label"] == 0).sum())
    val_abnormal = int((val_dataframe["label"] == 1).sum())

    print("Train normal:", train_normal)
    print("Train abnormal:", train_abnormal)
    print("Val normal:", val_normal)
    print("Val abnormal:", val_abnormal)


def main():
    make_binary_splits()


if __name__ == "__main__":
    main()