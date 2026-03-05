#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split




def load_imdb_split(folder):
    texts = []
    labels = []

    for label_type in ["pos", "neg"]:
        path = os.path.join(folder, label_type)

        for fname in os.listdir(path):
            if fname.endswith(".txt"):
                with open(os.path.join(path, fname), encoding="utf-8") as f:
                    texts.append(f.read())

                labels.append(1 if label_type == "pos" else 0)

    return texts, np.array(labels, dtype=np.float32)


base_dir = "./aclImdb"  # pasta onde está o dataset

train_texts, train_labels = load_imdb_split(os.path.join(base_dir, "train"))
test_texts, test_labels = load_imdb_split(os.path.join(base_dir, "test"))


max_words = 10000

vectorizer = CountVectorizer(
    max_features=max_words,
    binary=True,
    stop_words="english"
)

x_train = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
x_test = vectorizer.transform(test_texts).toarray().astype(np.float32)


x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    train_labels,
    test_size=10000,
    random_state=42,
    stratify=train_labels
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train = torch.tensor(x_train).to(device)
y_train = torch.tensor(y_train).unsqueeze(1).to(device)

x_val = torch.tensor(x_val).to(device)
y_val = torch.tensor(y_val).unsqueeze(1).to(device)

x_test = torch.tensor(x_test).to(device)
y_test = torch.tensor(test_labels).unsqueeze(1).to(device)




class IMDBModel(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(IMDBModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


model = IMDBModel(max_words, 128).to(device)

print(model)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



num_epochs = 10
batch_size = 512

for epoch in range(num_epochs):

    model.train()

    permutation = torch.randperm(x_train.size()[0])

    for i in range(0, x_train.size()[0], batch_size):

        indices = permutation[i:i + batch_size]

        batch_x = x_train[indices]
        batch_y = y_train[indices]

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():

        val_outputs = model(x_val)

        val_loss = criterion(val_outputs, y_val)

        val_preds = (val_outputs >= 0.5).float()

        val_acc = (val_preds == y_val).float().mean()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Val Loss: {val_loss.item():.4f} | "
        f"Val Acc: {val_acc.item():.4f}"
    )


model.eval()

with torch.no_grad():

    test_outputs = model(x_test)

    test_loss = criterion(test_outputs, y_test)

    test_preds = (test_outputs >= 0.5).float()

    test_acc = (test_preds == y_test).float().mean()

print("\nTest Results")
print("Test Loss:", test_loss.item())
print("Test Accuracy:", test_acc.item())