import json
import numpy as np
from nltk_utils import tokenization, stemming, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags =[]
xy =[]

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenization(pattern)
        all_words.extend(w)
        xy.append((w,tag))

punctuation = ['?', ',', '.', '!']

all_words = [stemming(word) for word in all_words if word not in punctuation]
all_words = sorted(set(all_words))
tags = sorted(set(tags))   # not necessary but for safty

X_train =[]
y_train = []

for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)  # converting tags into indecies e.g (0,1,2,3,...)
    y_train.append(label)

X_train = np.array(X_train) # converting them from list to array to be used in training
y_train = np.array(y_train)

#Hyperparameters tuning
batch_sizze = 8
input_size = len(all_words) # or len(bag)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# create a new dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

# dataset[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

# get number of samples
    def __len__(self):
        return self.n_samples


datasett = ChatDataset()
train_loader = DataLoader(dataset=datasett, batch_size=batch_sizze, shuffle=True, num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
critirion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        loss = critirion(outputs, labels)

        # backword
        optimizer.zero_grad()
        loss.backward()  # compute loss function in backward propagation
        optimizer.step()

    if (epoch+1)%100 == 0: #reachs 100 epochs
        print(f'epoch [{epoch+1}/{num_epochs}] , loss: {loss.item():.4f}')


print(f'final loss : {loss.item():.4f}')

# saving the model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

# defining a file to save data in it
FILE = "data.pth"
torch.save(data,FILE)

print(f"tarining completed, and file saved to {FILE}")