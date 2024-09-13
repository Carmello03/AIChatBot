import json
import numpy as np
from nltk_utils import tokenize, stemming, bag_of_words
from model import NeuralNet  # Assuming NeuralNet is in model.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load intents data
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)  # extend instead of append because we don't want array of arrays
        xy.append((w, tag))

ignore_words = ['?', "!", ",", "."]
all_words = [stemming(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))  # sort only unique words
tags = sorted(set(tags))

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  # Cross Entropy Loss

x_train = np.array(x_train)
y_train = np.array(y_train)

# Ensure the following block is protected to avoid multiprocessing errors on Windows
if __name__ == '__main__':

    # Dataset class
    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(x_train)
            self.x_train = x_train
            self.y_train = y_train

        # return dataset with index
        def __getitem__(self, index):
            return self.x_train[index], self.y_train[index]

        # return len of samples
        def __len__(self):
            return self.n_samples


    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(x_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device, dtype=torch.int64)

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1} / {num_epochs}, loss={loss.item():.4f}')

    print(f'Final loss: loss={loss.item():.5f}')

    data = {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'Training Complete, Data saved to {FILE}')
