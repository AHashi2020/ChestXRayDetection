import os
import pandas as pd


vindr_root = "/Users/Annmarie/Desktop/cv_final_code/vindr-cxr_test"
test_dir = os.path.join(vindr_root, "test")
annotations_dir = os.path.join(vindr_root, "annotations")

image_labels_csv = os.path.join(annotations_dir, "image_labels_test.csv")
annotations_csv = os.path.join(annotations_dir, "annotations_test.csv")

filtered_image_labels_csv = os.path.join(annotations_dir, "image_labels_test_subset.csv")
filtered_annotations_csv = os.path.join(annotations_dir, "annotations_test_subset.csv")


def strip_extension(file_name):
    lower_name = file_name.lower()

    if lower_name.endswith(".dicom"):
        return file_name[:-6]
    if lower_name.endswith(".dcm"):
        return file_name[:-4]

    return os.path.splitext(file_name)[0]


def get_downloaded_image_ids(folder_path):
    image_ids = set()

    for file_name in os.listdir(folder_path):
        lower_name = file_name.lower()

        if lower_name.endswith(".dicom") or lower_name.endswith(".dcm"):
            image_id = strip_extension(file_name)
            image_ids.add(image_id)

    return image_ids


def find_image_id_column(dataframe):
    for column_name in dataframe.columns:
        normalized = column_name.strip().lower().replace("_", "").replace(" ", "")
        if normalized == "imageid":
            return column_name

    raise ValueError("Could not find image_id column")


def main():
    downloaded_image_ids = get_downloaded_image_ids(test_dir)
    print("Downloaded images found:", len(downloaded_image_ids), flush=True)

    image_labels_df = pd.read_csv(image_labels_csv)
    image_id_column = find_image_id_column(image_labels_df)

    filtered_image_labels_df = image_labels_df[
        image_labels_df[image_id_column].astype(str).isin(downloaded_image_ids)
    ].copy()

    filtered_image_labels_df.to_csv(filtered_image_labels_csv, index=False)

    print("Original image_labels rows:", len(image_labels_df), flush=True)
    print("Filtered image_labels rows:", len(filtered_image_labels_df), flush=True)
    print("Saved:", filtered_image_labels_csv, flush=True)

    if os.path.exists(annotations_csv):
        annotations_df = pd.read_csv(annotations_csv)
        annotation_image_id_column = find_image_id_column(annotations_df)

        filtered_annotations_df = annotations_df[
            annotations_df[annotation_image_id_column].astype(str).isin(downloaded_image_ids)
        ].copy()

        filtered_annotations_df.to_csv(filtered_annotations_csv, index=False)

        print("Original annotations rows:", len(annotations_df), flush=True)
        print("Filtered annotations rows:", len(filtered_annotations_df), flush=True)
        print("Saved:", filtered_annotations_csv, flush=True)
    else:
        print("annotations_test.csv not found, skipping box annotation filtering.", flush=True)


if __name__ == "__main__":
    main()