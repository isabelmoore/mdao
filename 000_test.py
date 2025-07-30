
import os
import argparse
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
from matplotlib.colors import Normalize
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset

class FlexibleNeuralNet(nn.Module):
    def __init__(self, layer_sizes, activation_type="tanh", dropout=0.5):
        super(FlexibleNeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i != 0 and i < len(layer_sizes) - 2:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            if activation_type == "tanh":
                self.layers.append(nn.Tanh())
            elif activation_type == "ReLU":
                self.layers.append(nn.LeakyReLU())
            elif activation_type == "SiLU":
                self.layers.append(nn.SiLU())
            else:
                pass
            if i != 0 and i < len(layer_sizes) - 2:
                self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.view(x.size(0), 2, 120)  # ensure final shape

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        lstm_out, (h0, c0) = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out)
        fc_out = self.fc(lstm_out)
        out = fc_out.permute(0, 2, 1)  # Ensure correct shape: [batch_size, output_dim, seq_length]
        
        return out

class ModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_size=64, num_layers=2, activation_type="tanh", epochs=100, lr=0.001, batch_size=32, device='cuda:0', patience=5, l1_lambda=0.001, dropout=0.5, model_type='flex'):
        self.hidden_size = hidden_size
        self.num_layers = num_layers        
        self.activation_type = activation_type
        self.epochs = epochs
        self.lr = lr
        self.model = None 
        self.optimizer = None 
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.device = device
        self.patience = patience
        self.l1_lambda = l1_lambda
        self.dropout = dropout
        self.model_type = model_type

    def fit(self, X, y):
        """
        Train the neural network model.
        
        :param X: Input features as a NumPy array.
        :param y: Target values as a NumPy array.
        :return: Self.
        """
        if self.model_type == 'flex':
            input_size = X.shape[1]
            output_size = y.shape[1] * y.shape[2] if len(y.shape) > 2 else y.shape[1]
            output_size = 240
            layer_sizes = [input_size] + [self.hidden_size] * self.num_layers + [output_size]
            print(f"Params: activation={self.activation_type}, epochs={self.epochs}, lr={self.lr}, batch_size={self.batch_size}, patience={self.patience}")
            self.model = FlexibleNeuralNet(layer_sizes=layer_sizes, activation_type=self.activation_type, dropout=self.dropout).to(self.device)
            
        elif self.model_type == 'lstm':
            input_dim = X.shape[2] if len(X.shape) > 2 else 1
            output_dim = y.shape[2] if len(y.shape) > 2 else 1
            print(f"Params: epochs={self.epochs}, lr={self.lr}, batch_size={self.batch_size}, patience={self.patience}")
            self.model = LSTMModel(input_dim, output_dim, self.hidden_size, self.num_layers, self.dropout).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
    
        self.model.train()
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_loss = float('inf')
        patience_counter = 0
        epoch_loss = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                
                if self.model_type == 'lstm':
                    step_losses = []
                    for step in range(1, batch_X.size(1) + 1):
                        if batch_X.dim() == 3:
                            current_inputs = batch_X[:, :step, :]
                            current_targets = batch_y[:, :step, :].squeeze(-1)
                        else:  # Handle the case where batch_X is 2D
                            current_inputs = batch_X[:, :step].unsqueeze(-1)
                            current_targets = batch_y[:, :step].unsqueeze(-1)
                        
                        assert current_inputs.size(-1) == self.model.lstm.input_size, f"Input size mismatch: expected {self.model.lstm.input_size}, got {current_inputs.size(-1)}"

                        outputs = self.model(current_inputs)
                        
                        # Ensure current_targets and outputs shape align properly
                        aligned_targets = current_targets.unsqueeze(-1).permute(0, 2, 1)
                        if aligned_targets.size(1) != outputs.size(1):
                            aligned_targets = aligned_targets[:, :, -outputs.size(1):]

                        assert outputs.size() == aligned_targets.size(), f"Outputs and targets shape mismatch: {outputs.size()} vs {aligned_targets.size()}"

                        loss = self.criterion(outputs, aligned_targets)
                        step_losses.append(loss.item())
                        
                        loss.backward()
                        self.optimizer.step()

                    epoch_loss += sum(step_losses) / len(step_losses) if step_losses else 0
                
                else:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    
                    l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                    l1_loss = self.l1_lambda * l1_norm
                    loss += l1_loss

            epoch_loss /= len(data_loader)

            # Calculate average loss for the epoch
            epoch_loss /= len(data_loader)    

            # Check for early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"Epoch: {epoch+1}/{self.epochs}: Early stopping triggered at epoch", )
                break

        # Free up memory after each epoch
        torch.cuda.empty_cache()

        return self
    
    def predict(self, X):
        """
        Predict using the trained neural network model.
        
        :param X: Input features as a NumPy array.
        :return: Predicted values as a NumPy array.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy()
