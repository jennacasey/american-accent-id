import os
import json
from sklearn.model_selection import train_test_split

def load_data_from_folder(data_dir, label_mapping):
    transcriptions = []
    accent_labels = []

    # Iterate through each folder (one for each accent)
    for accent_folder in os.listdir(data_dir):
        accent_folder_path = os.path.join(data_dir, accent_folder)

        # Check if the entry is a directory
        if os.path.isdir(accent_folder_path):
            # Iterate through each file in the folder
            for filename in os.listdir(accent_folder_path):
                file_path = os.path.join(accent_folder_path, filename)

                # Skip directories and non-text files
                if os.path.isdir(file_path) or not filename.endswith(".txt"):
                    continue

                # Extract the accent label from the file name
                try:
                    # Assuming your file names are formatted as "sample_{label}_..."
                    label = filename.split('_')[-1].split('.')[0]
                    # Map the label to its encoded value using the provided label_mapping dictionary
                    encoded_label = label_mapping.get(f"Accent Label: {label}", -1)

                    if encoded_label != -1:
                        with open(file_path, 'r') as file:
                            # Read the transcription from the file
                            transcription = file.readline().strip()

                        # Append the transcription and encoded label to the lists
                        transcriptions.append(transcription)
                        accent_labels.append(encoded_label)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    return transcriptions, accent_labels

# Define your label mapping dictionary
label_mapping = {
    "Accent Label: Midland": 9,
    "Accent Label: InlandNorth": 3,
    "Accent Label: MidAtlantic": 5,
    "Accent Label: Southern": 8,
    "Accent Label: NYC": 6,
    "Accent Label: WesternNE": 12,
    "Accent Label: InlandSouth": 4,
    "Accent Label: Northern": 7,
    "Accent Label: WestPennsylvania": 10,
    "Accent Label: Florida": 2
}

# Load transcriptions and accent labels
transcriptions, accent_labels = load_data_from_folder('/Users/jennacasey/Desktop/label_accent/Data', label_mapping)

# Find the maximum number of symbols
max_symbols = max(len(sequence) for sequence in transcriptions)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(transcriptions, accent_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Number of samples in training set: {len(X_train)}")
print(f"Number of samples in validation set: {len(X_val)}")
print(f"Number of samples in testing set: {len(X_test)}")
print(f"Maximum number of symbols: {max_symbols}")

# Save the splits to separate files
with open('train_data.json', 'w') as json_file:
    json.dump({"transcriptions": X_train, "accent_labels": y_train}, json_file)

with open('val_data.json', 'w') as json_file:
    json.dump({"transcriptions": X_val, "accent_labels": y_val}, json_file)

with open('test_data.json', 'w') as json_file:
    json.dump({"transcriptions": X_test, "accent_labels": y_test}, json_file)

