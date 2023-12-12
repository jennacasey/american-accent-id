import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from load_data import load_data_from_folder

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

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Find the maximum sequence length in the training set
max_symbols = 359

X_train_padded = pad_sequences(X_train_seq, maxlen=max_symbols, padding='post')
X_val_padded = pad_sequences(X_val_seq, maxlen=max_symbols, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_symbols, padding='post')

torch.save(X_train_padded, 'X_train_padded.pt')
torch.save(y_train, 'y_train.pt')

# Load padded sequences
X_train_padded = torch.load('X_train_padded.pt')
y_train = torch.load('y_train.pt')

X_train_tensor = torch.LongTensor(X_train_padded)
y_train_tensor = torch.LongTensor(y_train)

class AccentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size):
        super(AccentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
        output = self.fc(lstm_out)
        return output

# Parameters
vocab_size = 1323
embedding_dim = 200  # Adjust as needed
hidden_dim = 256  # Adjust as needed
output_size = 12  # Number of accent classes

# Instantiate the model
model = AccentModel(vocab_size, embedding_dim, hidden_dim, output_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.LongTensor(X_train_padded)
y_train_tensor = torch.LongTensor(y_train)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_tensor)

    # Calculate loss
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.LongTensor(X_val_padded))
        val_loss = criterion(val_outputs, torch.LongTensor(y_val))
        print(f'Validation Loss: {val_loss.item():.4f}')

    # Add early stopping condition (you may adjust this condition)
    if epoch > 10 and val_loss > prev_val_loss:
        print("Early stopping.")
        break

    prev_val_loss = val_loss
