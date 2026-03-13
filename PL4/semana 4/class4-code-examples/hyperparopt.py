#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class FFNN(nn.Module):
    def __init__(self, nfeatures, nclasses = 2, topology = [50], algorithm = "adam", 
                 dropout = 0.0, learning_rate = 0.001, l2 = 0.0, early_stopping = -1): 
        ## topology - list with hidden layers and number of nodes in hidden layers
        ## dropout - dropout rate - if 0.0 no dropout
        ## l2 - l2 lamdba; ealry stopping patience (if <0 early stopping not used)
        super(FFNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = [nn.Flatten()]
        if topology != []:
            layers.append(nn.Linear(nfeatures, topology[0]))
            layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for i in range(1, len(topology)):
            layers.append (nn.Linear(topology[i-1], topology[i]))  
            layers.append (nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        if (nclasses>2): 
            layers.append(nn.Linear(topology[len(topology)-1], nclasses))
        else: 
            layers.append(nn.Linear(topology[len(topology)-1], 1))
            layers.append(nn.Sigmoid())
        model = nn.Sequential(*layers)
        self.net = model.to(self.device)
        self.set_loss(nclasses)
        self.set_optimizer(optimizer = algorithm, lr = learning_rate, l2 = l2)
    
        if early_stopping >= 0:
            self.set_early_stopping(early_stopping)
        else: self.early_stopping = None
            
    def forward(self, x):
        return self.net(x)
    
    def set_loss(self, nclasses):
        if(nclasses>2): 
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCELoss()
        
    def set_optimizer(self, optimizer = "adam", lr = 0.001, l2 = 0.0, momentum = 0.0):
        if optimizer == "adam": self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay = l2)
        elif optimizer == "sgd": self.optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay = l2)
        elif optimizer == "rmsprop": self.optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay = l2)
        elif optimizer == "sgdmomentum": self.optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay = l2, momentum = momentum)
        else: 
            optimizer = None
            print("Error ! Algorithm does not exist")
    
    def set_early_stopping(self, patience = 3, delta = 0.0):
        self.early_stopping = EarlyStopping(patience=patience, delta=delta)
    
    def train_model(self, train_loader, val_loader, epochs = 20, l1 = 0.0, verbose = False):
        network = self.net
        for epoch in range(epochs):
            # ---- Training ----
            network.train()
            correct = 0
            total = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
        
                outputs = network(images)
                loss = self.criterion(outputs, labels)
                
                if l1 > 0.0:
                    l1_norm = sum(p.abs().sum() for p in network.parameters())
                    loss += l1 * l1_norm
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
            train_acc = correct / total
        
            # ---- Validation ----
            network.eval()
            correct = 0
            total = 0
        
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
        
                    outputs = network(images)
                    val_loss = self.criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
        
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
        
            val_acc = correct / total
            
            if(verbose): print(f"Epoch {epoch+1:02d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | ValLoss: {val_loss:.4f} ")
            
            if self.early_stopping is not None:
                self.early_stopping(val_acc)
                ## could be: self.early_stopping(-val_loss)
                if self.early_stopping.early_stop:
                    if (verbose): print("Early stopping")
                    val_acc = self.early_stopping.best_score
                    break
    
        return val_acc


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, optimized_metric):
        score = optimized_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def dnn_optimization(opt_params, train_loader, val_loader, input_size, nclasses, iterations = 10, epochs = 5, 
                     early_stopping = -1, verbose = True, print_epochs = False):
    from random import choice
  
    if verbose: 
        print("Topology\tDropout\tAlgorithm\tLRate\tL2\tValAcc\n")
    best_acc = None
    
    if "topology" in opt_params:
        topologies = opt_params["topology"]
    else: topologies = [[100]]
    if "algorithm" in opt_params:
        algs = opt_params["algorithm"]
    else: algs = ["adam"]
    if "lr" in opt_params:
        lrs = opt_params["lr"]
    else: lrs = [0.001]
    if "dropout" in opt_params:
        dropouts = opt_params["dropout"]
    else: dropouts= [0.0]
    if "l2" in opt_params:
        l2s = opt_params["l2"]
    else: l2s = [0.001]
    
    for it in range(iterations):
        topo = choice(topologies)
        dropout_rate = choice(dropouts)
        alg = choice(algs)
        lr = choice(lrs)
        l2 = choice (l2s)
        dnn = FFNN(input_size, nclasses, topology = topo, algorithm = alg, 
                   dropout = dropout_rate, learning_rate = lr, l2 = l2, early_stopping= early_stopping)
        
        val_acc = dnn.train_model(train_loader, val_loader, verbose = print_epochs)
        
        if verbose: 
            print(topo, "\t", dropout_rate, "\t", alg, "\t", lr, "\t", l2, "\t", val_acc)
        
        if best_acc is None or val_acc > best_acc:
            best_acc = val_acc
            best_config = (topo, dropout_rate, alg, lr)
        
    return best_config, best_acc

def load_dataset_mnist():
    # Load the full dataset
    transform = transforms.ToTensor()
    
    full_train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [50000, 10000])
    
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return train_loader, val_loader, test_loader

## test
train_loader, val_loader, test_loader = load_dataset_mnist()

opt_pars = {"topology":[[100], [100,50], [250], [250,100]],
            "algorithm": [ "adam", "rmsprop"],
            "lr": [0.01, 0.001],
            "dropout": [0, 0.2], "l2": [0.001, 0.0]}

best_config, best_val_acc = dnn_optimization(opt_pars, train_loader, val_loader, input_size = 784, nclasses = 10, 
                                             iterations = 20, epochs = 20, early_stopping = 3, print_epochs = False)  
print("Best configuration:", best_config)
print("Best validation accuracy:", best_val_acc) 

