import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# Configure logging
logging.basicConfig(filename='accent_id.log', level=logging.DEBUG)

# Define your model
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = torch.mean(output, dim=1)  # Assuming you're working with sequences
        output = self.fc(output)
        return output

# Set your hyperparameters
vocab_size = 1323
embedding_dim = 100  # Adjust as needed
hidden_size = 128
output_size = 12

# Create an instance of your model
model = SimpleModel(vocab_size, embedding_dim, hidden_size, output_size)

# Set your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load your data
X_train_padded = torch.load('X_train_padded.pt')
y_train = torch.load('y_train.pt') - 1  # Subtract 1 to make the targets in the range [0, 11]

# Convert your data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_padded, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
epochs = 10  # Adjust as needed
for epoch in range(epochs):
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item()}')

    # Log progress
    logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'simple_model.pth')



        
