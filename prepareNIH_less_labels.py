import os
import random
import pandas as pd


# --------------------------------------------------
# Config
# --------------------------------------------------
nih_root = "NIH"
train_output_csv = "train_less_labels.csv"
val_output_csv = "val_less_labels.csv"

val_fraction = 0.20
random_seed = 42

original_label_names = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

grouped_label_map = {
    "Airspace_Opacity": [
        "Atelectasis",
        "Infiltration",
        "Consolidation",
        "Pneumonia",
        "Edema",
    ],
    "Cardiomediastinal_Abnormality": [
        "Cardiomegaly",
    ],
    "Pleural_Abnormality": [
        "Effusion",
        "Pneumothorax",
        "Pleural_Thickening",
    ],
    "Focal_Lesion": [
        "Mass",
        "Nodule",
    ],
    "Chronic_Parenchymal_Change": [
        "Emphysema",
        "Fibrosis",
    ],
}

grouped_label_names = [
    "Airspace_Opacity",
    "Cardiomediastinal_Abnormality",
    "Pleural_Abnormality",
    "Focal_Lesion",
    "Chronic_Parenchymal_Change",
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def find_metadata_csv():
    possible_paths = [
        "Data_Entry_2017.csv",
        os.path.join(nih_root, "Data_Entry_2017.csv"),
    ]

    path_index = 0
    while path_index < len(possible_paths):
        current_path = possible_paths[path_index]
        if os.path.exists(current_path):
            print("Found metadata CSV:", current_path, flush=True)
            return current_path
        path_index += 1

    raise FileNotFoundError(
        "Could not find Data_Entry_2017.csv in current folder or inside NIH/"
    )


def build_empty_original_label_dict():
    label_dict = {}
    label_index = 0

    while label_index < len(original_label_names):
        current_label = original_label_names[label_index]
        label_dict[current_label] = 0
        label_index += 1

    return label_dict


def parse_finding_labels(finding_string, unexpected_labels_set):
    label_dict = build_empty_original_label_dict()

    if pd.isna(finding_string):
        return label_dict

    findings_text = str(finding_string).strip()
    if findings_text == "":
        return label_dict

    parts = findings_text.split("|")

    part_index = 0
    while part_index < len(parts):
        current_part = parts[part_index].strip()

        if current_part == "" or current_part == "No Finding":
            part_index += 1
            continue

        if current_part in label_dict:
            label_dict[current_part] = 1
        else:
            unexpected_labels_set.add(current_part)

        part_index += 1

    return label_dict


def build_grouped_label_dict(original_label_dict):
    grouped_dict = {}

    grouped_index = 0
    while grouped_index < len(grouped_label_names):
        grouped_name = grouped_label_names[grouped_index]
        source_labels = grouped_label_map[grouped_name]

        grouped_value = 0
        source_index = 0
        while source_index < len(source_labels):
            source_label = source_labels[source_index]
            if original_label_dict[source_label] == 1:
                grouped_value = 1
                break
            source_index += 1

        grouped_dict[grouped_name] = grouped_value
        grouped_index += 1

    return grouped_dict


def build_image_path_map(root_folder):
    print("Scanning for PNG images under:", root_folder, flush=True)

    image_path_map = {}
    total_found = 0

    for current_root, _, file_names in os.walk(root_folder):
        file_index = 0
        while file_index < len(file_names):
            file_name = file_names[file_index]

            if file_name.lower().endswith(".png"):
                full_path = os.path.join(current_root, file_name)
                image_path_map[file_name] = full_path
                total_found += 1

            file_index += 1

    print("Total PNG images found:", total_found, flush=True)
    return image_path_map


def split_patient_ids(patient_ids, validation_fraction, seed):
    patient_id_list = []
    seen_ids = set()

    id_index = 0
    while id_index < len(patient_ids):
        current_id = patient_ids[id_index]
        if current_id not in seen_ids:
            seen_ids.add(current_id)
            patient_id_list.append(current_id)
        id_index += 1

    rng = random.Random(seed)
    rng.shuffle(patient_id_list)

    val_count = int(len(patient_id_list) * validation_fraction)

    if val_count <= 0 and len(patient_id_list) > 1:
        val_count = 1

    val_ids = set()
    train_ids = set()

    index = 0
    while index < len(patient_id_list):
        current_id = patient_id_list[index]

        if index < val_count:
            val_ids.add(current_id)
        else:
            train_ids.add(current_id)

        index += 1

    return train_ids, val_ids


def summarize_dataframe(dataframe, name):
    print("", flush=True)
    print(name + " summary", flush=True)
    print("Rows:", len(dataframe), flush=True)

    if len(dataframe) == 0:
        return

    label_frame = dataframe[grouped_label_names]
    normal_mask = label_frame.sum(axis=1) == 0
    normal_count = int(normal_mask.sum())
    abnormal_count = int(len(dataframe) - normal_count)

    print("Normal images:", normal_count, flush=True)
    print("Abnormal images:", abnormal_count, flush=True)

    label_index = 0
    while label_index < len(grouped_label_names):
        current_label = grouped_label_names[label_index]
        positive_count = int(dataframe[current_label].sum())
        print(current_label + " positives:", positive_count, flush=True)
        label_index += 1


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Preparing NIH grouped-label CSVs...", flush=True)

    metadata_csv_path = find_metadata_csv()
    metadata_df = pd.read_csv(metadata_csv_path)
    print("Loaded metadata rows:", len(metadata_df), flush=True)

    image_path_map = build_image_path_map(nih_root)

    records = []
    missing_images = 0
    unexpected_labels = set()

    row_index = 0
    while row_index < len(metadata_df):
        row = metadata_df.iloc[row_index]

        image_name = str(row["Image Index"]).strip()

        if image_name not in image_path_map:
            missing_images += 1
            row_index += 1
            continue

        patient_id = row["Patient ID"]
        finding_string = row["Finding Labels"]

        original_label_dict = parse_finding_labels(finding_string, unexpected_labels)
        grouped_label_dict = build_grouped_label_dict(original_label_dict)

        record = {}
        record["image_path"] = image_path_map[image_name]
        record["image_name"] = image_name
        record["patient_id"] = patient_id

        label_index = 0
        while label_index < len(grouped_label_names):
            current_label = grouped_label_names[label_index]
            record[current_label] = grouped_label_dict[current_label]
            label_index += 1

        records.append(record)

        if (row_index + 1) % 20000 == 0:
            print(
                "Processed",
                row_index + 1,
                "/",
                len(metadata_df),
                "metadata rows...",
                flush=True,
            )

        row_index += 1

    full_df = pd.DataFrame(records)

    print("", flush=True)
    print("Finished building grouped-label table.", flush=True)
    print("Rows kept:", len(full_df), flush=True)
    print("Missing images:", missing_images, flush=True)

    if len(unexpected_labels) > 0:
        print("Unexpected labels found in metadata:", flush=True)
        for label_value in sorted(unexpected_labels):
            print(" -", label_value, flush=True)

    if len(full_df) == 0:
        raise ValueError("No valid rows were created. Check NIH path and metadata CSV.")

    patient_ids = []
    unique_patient_series = full_df["patient_id"].drop_duplicates()

    series_index = 0
    while series_index < len(unique_patient_series):
        patient_ids.append(unique_patient_series.iloc[series_index])
        series_index += 1

    print("Unique patients found:", len(patient_ids), flush=True)

    train_patient_ids, val_patient_ids = split_patient_ids(
        patient_ids,
        val_fraction,
        random_seed,
    )

    train_df = full_df[full_df["patient_id"].isin(train_patient_ids)].copy()
    val_df = full_df[full_df["patient_id"].isin(val_patient_ids)].copy()

    column_order = ["image_path", "image_name", "patient_id"]

    label_index = 0
    while label_index < len(grouped_label_names):
        column_order.append(grouped_label_names[label_index])
        label_index += 1

    train_df = train_df[column_order]
    val_df = val_df[column_order]

    train_df.to_csv(train_output_csv, index=False)
    val_df.to_csv(val_output_csv, index=False)

    print("", flush=True)
    print("Saved:", train_output_csv, flush=True)
    print("Saved:", val_output_csv, flush=True)

    summarize_dataframe(train_df, "Train")
    summarize_dataframe(val_df, "Validation")

    print("", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()