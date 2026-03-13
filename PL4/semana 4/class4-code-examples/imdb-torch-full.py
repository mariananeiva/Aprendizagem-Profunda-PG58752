# -*- coding: utf-8 -*-

import os
import re
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

## FUNCTIONS FOR TEXT PROCESSING 

def clean_text(text): ## standardization
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def build_vocab(texts, max_words = 10000): ## building the vocabulary
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    most_common = counter.most_common(max_words)
    word_index = {word: i for i, (word, _) in enumerate(most_common)}
    return word_index

def vectorize_text(text, word_index, max_words): 
    ## one hot encoding (simply presence/ absence of word as feature)
    vector = torch.zeros(max_words)
    for word in set(text.split()): ## assuming text is already clean in load_split
        if word in word_index:
            vector[word_index[word]] = 1
    return vector

def encode(text, word_index, max_len=200): ## integer encoding
    tokens = clean_text(text).split()
    
    sequence = [word_index.get(word, 1) for word in tokens]
    # Truncate
    sequence = sequence[:max_len]
    # Pad
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))

    return torch.tensor(sequence, dtype=torch.long)

## Loading data / creating datasets
def load_split(data_dir, split):
    texts = []
    labels = []

    for label_type in ['pos', 'neg']:
        dir_path = os.path.join(data_dir, split, label_type)

        for fname in os.listdir(dir_path):
            with open(os.path.join(dir_path, fname), encoding='utf8') as f:
                texts.append(clean_text(f.read()))
                labels.append(1 if label_type == 'pos' else 0)

    return texts, labels

class IMDBDataset(Dataset): ## dataset with one-hot encoding
    def __init__(self, texts, labels, word_index, max_words):
        self.texts = texts
        self.labels = labels
        self.word_index = word_index
        self.max_words = max_words

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = vectorize_text(self.texts[idx], self.word_index, self.max_words)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

class TfidfDataset(Dataset): ## dataset with TF-IDF features
    def __init__(self, texts, labels, max_words = 20000, train = True, tr_texts = None):
        self.texts = texts
        self.labels = labels
        self.vectorizer = TfidfVectorizer(max_features=max_words, ngram_range=(1, 2), stop_words="english", min_df=5)
        if train: 
            self.X = torch.tensor(self.vectorizer.fit_transform(texts).toarray(), dtype=torch.float32)
        else:
            self.vectorizer.fit(tr_texts)
            self.X = torch.tensor(self.vectorizer.transform(texts).toarray(), dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

class IMDBDatasetEmbed(Dataset): 
    ## dataset with integer indexes (for embedding/ RNNs and so on)
    def __init__(self, texts, labels, word_index, max_words, max_len =200):
        self.texts = texts
        self.labels = labels
        self.word_index = word_index
        self.max_words = max_words
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode(self.texts[idx], self.word_index, self.max_len)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

## Loading datasets for the different algorithms
## one-hot encoding
def load_dataset(filespath, max_words = 10000, batch_size = 512, val_perc = 0.8):
    ## load splits
    train_texts, train_labels = load_split(filespath, "train")
    test_texts, test_labels = load_split(filespath, "test")
    
    ## build vocabulary
    word_index = build_vocab(train_texts, max_words)
    ##print(word_index)  ## uncomment to check the vocabulary
    
    ## create train and test splits
    full_train_dataset = IMDBDataset(train_texts, train_labels, word_index, max_words)
    test_dataset = IMDBDataset(test_texts, test_labels, word_index, max_words)
    
    train_size = int(val_perc * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size ## 20% for validation
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducibility
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

## TF-IDF dataset
def load_dataset_tfidf(filespath, max_words =10000, batch_size = 256, val_perc = 0.8):

    ## load splits
    train_texts, train_labels = load_split(filespath, "train")
    test_texts, test_labels = load_split(filespath, "test")
    
    ## create train and test splits
    full_train_dataset = TfidfDataset(train_texts, train_labels, max_words, True)
    test_dataset = TfidfDataset(test_texts, test_labels, max_words, False, train_texts)
    
    train_size = int(val_perc * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size ## 20% for validation
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducibility
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

## dataset for embeddings, RNNs, LSTMs, etc
def load_dataset_embed(filespath, max_words = 10000, max_len = 200, batch_size = 512, val_perc = 0.8):
    ## load splits
    train_texts, train_labels = load_split(filespath, "train")
    test_texts, test_labels = load_split(filespath, "test")
    
    ## build vocabulary
    word_index = build_vocab(train_texts, max_words)
    
    ## create train and test splits
    full_train_dataset = IMDBDatasetEmbed(train_texts, train_labels, word_index, max_words, max_len)
    test_dataset = IMDBDatasetEmbed(test_texts, test_labels, word_index, max_words, max_len)
    
    train_size = int(val_perc * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size ## 20% for validation
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducibility
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

## Pre-trained embeddings
def load_glove_embeddings(glove_path, word_index, embed_dim):
    embeddings_index = {}

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    vocab_size = len(word_index)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))

    # PAD token = zeros
    embedding_matrix[0] = np.zeros(embed_dim)

    found = 0
    for word, i in word_index.items():
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
            found += 1

    print(f"Found {found}/{vocab_size} pretrained vectors")

    return torch.tensor(embedding_matrix, dtype=torch.float32)

## Model training

def train(model, train_loader, val_loader, criterion, epochs = 5, lr = 0.001, verbose = True):
    ## verbose - print losses and accuracies per epoch
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs): 
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if verbose: 
            print(f"Epoch {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
    
    return train_accs, val_accs, train_losses, val_losses

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total

## MODELS

## Logistic regression
class SimpleClassifier(nn.Module):  
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()

## FFNN
class FFNN(nn.Module):
    def __init__(self, input_dim, topology = [50], dropout = 0.0): 
        ## topology - list with hidden layers and number of nodes in hidden layers
        ## dropout - dropout rate - if 0.0 no dropout

        super(FFNN, self).__init__()
        layers = []
        if not topology:
            layers.append(nn.Linear(input_dim, 1))
        else:
            layers.append(nn.Linear(input_dim, topology[0]))
            layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for i in range(1, len(topology)):
            layers.append(nn.Linear(topology[i-1], topology[i]))  
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(topology[len(topology)-1], 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze()

## Embeddings
class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, 1)
        print(self.embedding.weight.numel())

    def forward(self, x): # x shape: (batch_size, max_len)
        embedded = self.embedding(x)
        # shape: (batch_size, max_len, embed_dim)
        pooled = embedded.mean(dim=1)
        # shape: (batch_size, embed_dim)
        output = self.fc(pooled)
        return output.squeeze(1)

class EmbeddingPretrained(nn.Module):
    def __init__(self, glove_path, word_index, embed_dim):
        super().__init__()
        embedding_matrix = load_glove_embeddings(glove_path, word_index, embed_dim)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze= False,
            padding_idx=0
        )
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):   # x shape: (batch_size, max_len)
        embedded = self.embedding(x)
        # shape: (batch_size, max_len, embed_dim)
        pooled = embedded.mean(dim=1)
        # shape: (batch_size, embed_dim)
        output = self.fc(pooled)
        return output.squeeze(1)

## RNNs
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)   # (B, L, E)
        output, hidden = self.rnn(embedded)  # output: (B, L, H); hidden: (num_layers, B, H)
        last_hidden = hidden[-1]   # (B, H)
        out = self.fc(last_hidden)
        return out.squeeze(1)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.0, bidirectional = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional = bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, 1)

    def forward(self, x): # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)    # (B, L, E)
        output, (hidden, cell) = self.lstm(embedded)   # hidden: (num_layers, B, hidden_dim)
        
        if self.lstm.bidirectional:
            # last forward + last backward
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            last_hidden = hidden[-1] # Take hidden state from last LSTM layer
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out.squeeze(1)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.0, bidirectional = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, 1)

    def forward(self, x):     # x: (batch_size, seq_len)
        embedded = self.embedding(x)    # (B, L, E)
        output, hidden = self.gru(embedded)
        # hidden: (num_layers * directions, B, hidden_dim)
        if self.gru.bidirectional:
            # last forward + last backward
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            last_hidden = hidden[-1]
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out.squeeze(1)        

### plots

def plot_values(list_values, list_labels  = ["Train", "Validation"], ylabel = "Accuracy", title = None):
    plt.figure()
    for i in range(len(list_values)):
        plt.plot(list_values[i], label = list_labels[i])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    if title is not None: plt.title()
    plt.show()

#### TESTS #####

def test_log_reg():
    filespath = "../class3-code/aclImdb"
    max_words = 20000
    train_loader, val_loader, test_loader = load_dataset(filespath, max_words)
    model = SimpleClassifier(max_words)
    criterion = nn.BCELoss()
    train_accs, val_accs, train_losses, val_losses = train(model, train_loader, val_loader, criterion, epochs = 20)
    
    plot_values([train_accs, val_accs])
    plot_values([train_losses, val_losses], ylabel = "Loss")
    
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

def test_ffnn():
    filespath = "../class3-code/aclImdb"
    max_words = 20000
    train_loader, val_loader, test_loader = load_dataset(filespath, max_words)
    model = FFNN(max_words, topology = [100], dropout = 0.5)
    criterion = nn.BCELoss()
    train(model, train_loader, val_loader, criterion, epochs = 10)
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

def test_tf_idf():
    filespath = "../class3-code/aclImdb"
    max_words = 20000
    train_loader, val_loader, test_loader = load_dataset_tfidf(filespath, max_words)
    #model = SimpleClassifier(max_words) ## logistic regression
    model = FFNN(max_words, topology = [100], dropout = 0.2) ## DNN
    criterion = nn.BCELoss()
    train(model, train_loader, val_loader, criterion, epochs = 20)
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

def test_embed():
    filespath = "../class3-code/aclImdb"
    max_words = 20000
    max_len = 200
    embed_dim = 200
    train_loader, val_loader, test_loader = load_dataset_embed(filespath, max_words, max_len = max_len)
    model = EmbeddingClassifier(max_words, embed_dim)
    criterion = nn.BCEWithLogitsLoss()
    train(model, train_loader, val_loader, criterion, epochs = 20)
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

def test_embed_pretr():
    filespath = "../class3-code/aclImdb"
    glovepath = "glove.6B.100d.txt"
    max_words = 20000
    max_len = 200
    embed_dim = 100 ## the dimensions of glove
    train_texts, train_labels = load_split(filespath, "train")
    word_index = build_vocab(train_texts, max_words)
    train_loader, val_loader, test_loader = load_dataset_embed(filespath, max_words, max_len = max_len)
    model = EmbeddingPretrained(glovepath, word_index, embed_dim)
    criterion = nn.BCEWithLogitsLoss()
    train(model, train_loader, val_loader, criterion, epochs = 20)
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

def test_rnn():
    filespath = "../class3-code/aclImdb"
    max_words = 20000
    max_len = 100
    embed_dim = 200
    train_loader, val_loader, test_loader = load_dataset_embed(filespath, max_words, max_len = max_len)
    model = RNNClassifier(max_words,embed_dim, hidden_dim=128, num_layers=1)
    criterion = nn.BCEWithLogitsLoss()
    train(model, train_loader, val_loader, criterion, epochs = 20)
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

def test_lstm():
    filespath = "../class3-code/aclImdb"
    max_words = 20000
    max_len = 100 ## do not increase
    embed_dim = 200
    train_loader, val_loader, test_loader = load_dataset_embed(filespath, max_words, max_len = max_len)
    model = LSTMClassifier(max_words,embed_dim, hidden_dim=128, num_layers=2, bidirectional = True, dropout = 0.5)
    criterion = nn.BCEWithLogitsLoss()
    train(model, train_loader, val_loader, criterion, epochs = 20) #, lr = 0.0001)
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

def test_gru():
    filespath = "../class3-code/aclImdb"
    max_words = 20000
    max_len = 100 ## do not increase
    embed_dim = 200
    train_loader, val_loader, test_loader = load_dataset_embed(filespath, max_words, max_len = max_len)
    model = GRUClassifier(max_words,embed_dim, hidden_dim=128, num_layers=1, bidirectional = True, dropout = 0.3)
    criterion = nn.BCEWithLogitsLoss()
    train_accs, val_accs, train_losses, val_losses = train(model, train_loader, val_loader, criterion, epochs = 20) #, lr = 0.0001)
    
    plot_values([train_accs, val_accs])
    plot_values([train_losses, val_losses], ylabel = "Loss")
    
    _, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

#test_log_reg()
#test_ffnn()
#test_tf_idf()
#test_embed()
#test_embed_pretr()
#test_rnn()  
test_lstm()
#test_gru()