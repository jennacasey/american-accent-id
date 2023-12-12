import os
import json
from sklearn.preprocessing import LabelEncoder

def load_labels_from_files(data_dir, label_mapping):
    encoded_labels = {}
    label_encoder = LabelEncoder()

    for subdir, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(subdir, file)

            # Check if the file is a text file
            if file.endswith(".txt"):
                with open(file_path, 'r') as txt_file:
                    # Extract the accent label from the file name
                    accent_label = file.split('_')[-1][:-4]

                    try:
                        # Use the label mapping to get the numerical label
                        numerical_label = label_mapping[accent_label]
                        encoded_labels[file] = numerical_label
                    except KeyError:
                        print(f"KeyError: {accent_label} not found in label mapping for file {file}.")
                        print(f"File path: {file_path}")
                        print(f"File content: {txt_file.read().strip()}")
                        print("\n")

    return encoded_labels

def save_label_mapping(encoded_labels, output_file='encoded_labels.json'):
    # Convert int64 values to Python integers
    encoded_labels = {key: int(value) for key, value in encoded_labels.items()}

    with open(output_file, 'w') as json_file:
        json.dump(encoded_labels, json_file)

if __name__ == "__main__":
    data_directory = '/Users/jennacasey/Desktop/label_accent'
    
    # Replace this with your actual label mapping
    label_mapping = {
        'Midland': 9,
        'InlandNorth': 3,
        'MidAtlantic': 5,
        'Southern': 8,
        'NYC': 6,
        'WesternNE': 12,
        'Western': 11,
        'EasternNE': 1,
        'InlandSouth': 4,
        'Northern': 7,
        'WestPennsylvania': 10,
        'Florida': 2,
    }

    encoded_labels = load_labels_from_files(data_directory, label_mapping)
    save_label_mapping(encoded_labels)

