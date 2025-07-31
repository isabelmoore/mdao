
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
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, seq_length, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = int(seq_length)  # Ensure seq_length is an integer
        self.input_dim = input_dim
        self.output_dim = int(output_dim)  # Ensure output_dim is an integer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x, h0=None, c0=None):
        if x.dim() != 3:
            raise ValueError(f"Unexpected input tensor shape: {x.shape}")
        
        # Initialize hidden and cell states if not provided
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        lstm_out, (h_t, c_t) = self.lstm(x, (h0, c0))
        fc_out = self.fc(lstm_out[:, -1, :])  # (batch_size, output_dim)
        fc_out_expanded = fc_out.unsqueeze(1).expand(-1, self.seq_length, -1)  # (batch_size, seq_length, output_dim)        
        return fc_out_expanded, (h_t, c_t)


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
        self.input_dim = 2
        self.output_dim = 2
        self.seq_length = 120
    def sliding_window(self, sequence, window_size):
        batch_size, seq_length, input_dim = sequence.shape
        num_windows = seq_length - window_size + 1

        # Ensure num_windows is non-negative
        if num_windows <= 0:
            raise ValueError("Invalid number of windows: {num_windows}. Check sequence length and window size.")

        windows = torch.zeros(batch_size, num_windows, window_size, input_dim, device=sequence.device)
        for i in range(num_windows):
            windows[:, i, :, :] = sequence[:, i:i + window_size, :]

        return windows
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
            self.model = FlexibleNeuralNet(layer_sizes=layer_sizes, activation_type=self.activation_type, Tp=self.dropout).to(self.device)
            
        if self.model_type == 'lstm':
            input_dim = 2
            output_dim = 2
            seq_length = 120
            print(f"Params: epochs={self.epochs}, lr={self.lr}, batch_size={self.batch_size}, patience={self.patience}")
            self.model = LSTMModel(input_dim, output_dim, self.hidden_size, self.num_layers, seq_length).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)

        self.model.train()
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_loss = float('inf')
        patience_counter = 0
        window_size = 10

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                print(f"Initial batch_X shape: {batch_X.shape}")

                # Initialize augmented input for the window size
                augmented_input = torch.zeros(batch_X.size(0), window_size, input_dim).to(self.device)

                # Fill in initial input with batch_X values for the initial window
                window_input = batch_X.unsqueeze(1).repeat(1, window_size, 1)  # Start with the first value and repeat for the window
                
                predictions = []
                h_t, c_t = None, None

                for t in range(window_size, seq_length):
                    out_t, (h_t, c_t) = self.model(window_input, h_t, c_t)
                    buffer = out_t[:, -1, :]  # Store the last prediction in the buffer

                    # Append the buffer (current prediction) to the predictions list
                    predictions.append(buffer.unsqueeze(1))

                    # Update the input window - Shift left and add the new prediction
                    window_input = torch.cat((window_input[:, 1:, :], buffer.unsqueeze(1)), dim=1)

                    print(f"Window input shape: {window_input.shape}")
                    print(f"Output shape: {out_t.shape}")

                # Concatenate all predictions along the sequence dimension
                predictions = torch.cat(predictions, dim=1)  # Shape: [batch_size, seq_length - window_size, output_dim]

                print(f"Predictions shape: {predictions.shape}")

                # Pad the initial part with the initial window_size predictions for proper shape
                predictions = torch.cat([augmented_input, predictions], dim=1)

                print(f"Padded Predictions shape: {predictions.shape}")
                print(f"Ground truth shape: {batch_y.shape}")

                # Ensure tensor shape compatibility
                predictions = predictions.permute(0, 2, 1)  # Transpose to match [batch_size, output_dim, seq_length]

                # Debugging checks for NaNs
                if torch.any(torch.isnan(predictions)):
                    raise ValueError("NaNs found in predictions")
                if torch.any(torch.isnan(batch_y)):
                    raise ValueError("NaNs found in targets")

                loss = self.criterion(predictions, batch_y)
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                l1_loss = self.l1_lambda * l1_norm
                loss += l1_loss
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(data_loader)  

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Epoch: {epoch+1}/{self.epochs}: Early stopping triggered at epoch {epoch+1}")
                break

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
            # Ensure proper input dimension [batch_size, input_dim]
            X_tensor = torch.from_numpy(X).float().to(self.device)
            batch_size = X_tensor.shape[0]
            predictions = torch.zeros(batch_size, self.output_dim, self.seq_length).to(self.device)
            input_seq = X_tensor.unsqueeze(1)  # Add sequence dimension [batch_size, 1, input_dim]
            h_t, c_t = None, None
            print(f"Initial input_seq shape for prediction: {input_seq.shape}")

            for t in range(self.seq_length):
                out_t, (h_t, c_t) = self.model(input_seq, h_t, c_t)  # Model output for the current time step
                predictions[:, :, t] = out_t[:, t, :]  # Store the latest prediction

                # Update input sequence to include the past predictions
                if t + 1 < self.seq_length:
                    input_seq = torch.cat((input_seq, out_t[:, t, :].unsqueeze(1)), dim=1)
                    # print(f"Updated input_seq.shape at step {t + 1}: {input_seq.shape}")

            return predictions.cpu().numpy()
