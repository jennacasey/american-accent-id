import os
from load_data import load_data_from_folder  # Replace with the actual module containing your data loading function


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

# Specify the path to your data directory
data_dir = '/Users/jennacasey/Desktop/label_accent/Data'

# Load transcriptions and accent labels
transcriptions, _ = load_data_from_folder(data_dir, label_mapping)

# Continue with the rest of the script

from keras.preprocessing.text import Tokenizer

# Assuming you have loaded your transcriptions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(transcriptions)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for the padding token
print(f"Vocabulary size: {vocab_size}")
