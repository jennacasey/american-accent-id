import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from load_data import load_data_from_folder
from mod_train import AccentModel  # Replace 'your_model_module' with the actual module containing your model definition

data_dir = '/Users/jennacasey/Desktop/label_accent/Data'

label_mapping = {
    "Accent Label: Midland": 9,
    "Accent Label: EasternNE": 1,
    "Accent Label: Western": 11,
    "Accent Label: InlandNorth": 3,
    "Accent Label: MidAtlantic": 5,
    "Accent Label: Southern": 8,
    "Accent Label: NYC": 6,
    "Accent Label: WesternNE": 12,
    "Accent Label: InlandSouth": 4,
    "Accent Label: Northern": 7,
    "Accent Label: WestPennsylvania": 10,
    "Accent Label: Florida": 2,
}

# Load transcriptions, accent labels, and sequence lengths
transcriptions, accent_labels, sequence_lengths = load_data_from_folder(data_dir, label_mapping)
accent_labels = [label - 1 for label in accent_labels]

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp, seq_lengths_train, seq_lengths_temp = train_test_split(
    transcriptions, accent_labels, sequence_lengths, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test, seq_lengths_val, seq_lengths_test = train_test_split(
    X_temp, y_temp, seq_lengths_temp, test_size=0.5, random_state=42)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Find the maximum sequence length in the training set
max_symbols = 359

X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_symbols, padding='post')

# Load your trained accent model
model = AccentModel(vocab_size=1323, embedding_dim=200, hidden_dim=256, output_size=12)
model.load_state_dict(torch.load('accent_model.pth'))  # Load the trained weights
model.eval()

# Create DataLoader for the test set
test_dataset = TensorDataset(torch.LongTensor(X_test_padded), torch.LongTensor(y_test))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Evaluate the model on the test set
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')


