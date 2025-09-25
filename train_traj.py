
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
        # Flatten the input for the fully connected layers
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x.view(x.size(0), 2, -1)  # Assure shape compatibility

class LSTMModel_OnetoMany(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, dropout=0.5):
        super(LSTMModel_OnetoMany, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = 2
        self.output_dim = 2
        self.output_length = 106        
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.final_output = nn.Conv1d(
            in_channels=self.hidden_size,  
            out_channels=self.output_dim * self.output_length,  
            kernel_size=1  
        )
        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        lstm_out = lstm_out.permute(0, 2, 1)  
        output = self.final_output(lstm_out) 
        output = output.view(-1, self.output_dim, self.output_length)
        return output
    
class LSTMModel_ManytoOne(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, dropout=0.5):
        super(LSTMModel_ManytoOne, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.final_output = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        lstm_out = lstm_out[:, -1, :] 
        output = self.final_output(lstm_out) 
        return output 

class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, nhead, max_len):
        super(TransformerSeq2Seq, self).__init__()
        # Transformer layers
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_size, nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        
        self.input_fc = nn.Linear(input_dim, hidden_size)  
        self.output_fc = nn.Linear(hidden_size, output_dim) 
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.input_fc(src).permute(1, 0, 2) 
        tgt = self.input_fc(tgt).permute(1, 0, 2)  
        
        memory = self.encoder(src, src_key_padding_mask=src_mask) 
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask) 
        
        return self.output_fc(output.permute(1, 0, 2))  

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
        self.window_size = 10

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
            layer_sizes = [input_size] + [self.hidden_size] * self.num_layers + [output_size]
            print(f"Params: activation={self.activation_type}, epochs={self.epochs}, lr={self.lr}, batch_size={self.batch_size}, patience={self.patience}")
            self.model = FlexibleNeuralNet(layer_sizes=layer_sizes, activation_type=self.activation_type, dropout=self.dropout).to(self.device)
            
        if self.model_type == 'lstm_onetomany':
            input_dim = 2
            output_dim = 2
            print(f"Params: epochs={self.epochs}, lr={self.lr}, batch_size={self.batch_size}, patience={self.patience}")
            self.model = LSTMModel_OnetoMany(input_dim, output_dim, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)
        seq_length = y_tensor.size(2) 

        self.model.train()
        dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            train_loss = 0
            self.model.train()
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                batch_X = batch_X.unsqueeze(1)

                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.model.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()  
                    
                train_loss += loss.item()
            train_loss /= len(data_loader)        

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Epoch: {epoch+1}/{self.epochs}: Early stopping triggered at epoch {epoch+1}")
                break

        torch.cuda.empty_cache()
        return self

    
    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        input_data = X
        batch_X_test = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(batch_X_test)
            predictions = predictions.cpu().numpy().reshape(-1, 111)
        
        return predictions




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

class NN_Functions():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)

        self.lowest_azimuth = 0
        self.lowest_range = 0
        self.window_size = 106

    def predicting(self, azimuth, range_, scaler_x, scaler_y):
        self.model.eval()  # Set the model to evaluation mode
        
        input_data = np.array([[azimuth, range_]], dtype=np.float32)
        scaled_input = scaler_x.transform(input_data)
        batch_X_test = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(1).to(self.device)
        model_type = self.model.__class__.__name__ 

        with torch.no_grad():
            if model_type == 'FlexibleNeuralNet' or model_type == 'LSTMModel_OnetoMany':
                predictions = self.model(batch_X_test)
                predictions_cpu = predictions.cpu().numpy().reshape(-1, 106)
                
                inverse_predictions = scaler_y.inverse_transform(predictions_cpu) 
                return inverse_predictions[0], inverse_predictions[1]


            elif model_type == "LSTMModel_ManytoOne":
                seq_length = 10
                window_size = 10  
                batch_size = 1   

                if len(batch_X_test.shape) == 3:
                    batch_X_test = batch_X_test.squeeze(1)  
                
                print(f"batch_X_test: {batch_X_test.shape}")

                predictions = []

                for start in range(seq_length + 1):
                    end = start + window_size

                    if end > seq_length:
                        end = seq_length

                    if start == 0:
                        # Initialize window_y to zeros for the first iteration
                        window_y = torch.zeros(1, 2, window_size).to(self.device)
                    else:
                        # Ensure the shape of window_y for consistency
                        if window_y is None or window_y.shape != (1, 2, window_size):
                            window_y = torch.zeros(1, 2, window_size).to(self.device)

                    window_X = batch_X_test.view(1, 2).unsqueeze(2).expand(-1, -1, window_size)  

                    window_input = torch.cat((window_X, window_y), dim=1)  # Shape: [1, 10, 4]
                    window_input = window_input.permute(0, 2, 1)  # Shape: [1, 4, 10] to [1, 10, 4]
                    # Model prediction
                    prediction = self.model(window_input)  # Shape: [1, 2]

                    # Update `window_y` for the next iteration
                    if start < seq_length:
                        window_y = torch.cat((window_y[:, :, 1:], prediction.unsqueeze(2)), dim=2)  # Keep size [64, 2, 10]
                    
                    prediction = prediction.unsqueeze(1)  # Shape: [1, 1, 2]
                    predictions.append(prediction)

                predictions_tensor = torch.cat(predictions, dim=1)  # Shape: [1, seq_length + 1, 2]
                print("predictions tensor:", predictions_tensor.shape)  # Should be [1, seq_length, 2]
                print("scaler:", scaler_y.scale_.shape)  # Should match the number of features, 2 in this case
                
                predictions_tensor_reshaped = predictions_tensor.cpu().numpy().reshape(-1, 2)  # Shape: [N, 2]

                # Now perform the inverse transform
                inverse_predictions = scaler_y.inverse_transform(predictions_tensor_reshaped)
                return inverse_predictions[:, 0], inverse_predictions[:, 1]

    ###### Loss vs Epochs Plot ######
    def plot_loss_vs_epoch(self, train_losses, test_losses):
        plt.figure(figsize=(12,8))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(test_losses, label="Testing Loss")
        plt.title(f"Loss over Epochs\nModel:{self.model.__class__.__name__}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig('loss_plot.png')  

    ###### MSE Polar Plot #######
    def plot_mse_polar(self, X_df, scaler_x, scaler_y):
        mse_values = []
        print(X_df['azimuth'][0:20])
        for index, row in X_df.iterrows():
            azimuth = row['azimuth']
            range_ = row['range']
            predicted_alpha, predicted_bank = self.predicting(azimuth, range_, scaler_x, scaler_y)
            
            if predicted_alpha is None or predicted_bank is None:
                print(f"Skipping index {index} due to prediction failure")
                continue
            
            actual_alpha = np.array(row['alpha_list'])[0]
            actual_bank = np.array(row['bank_list'])[0]
            alpha_mse = np.mean((predicted_alpha - actual_alpha) ** 2)
            bank_mse = np.mean((predicted_bank - actual_bank) ** 2)
            mse_values.append((alpha_mse, bank_mse))
        
        if len(mse_values) == 0:
            print("Error: mse_values is empty.")
            return

        azimuths = np.array(X_df['azimuth'])
        ranges = np.array(X_df['range'])
        alpha_mse_values = [mse[0] for mse in mse_values]
        bank_mse_values = [mse[1] for mse in mse_values]

        alpha_norm = Normalize(vmin=min(alpha_mse_values), vmax=max(alpha_mse_values))
        alpha_colors = plt.cm.viridis(alpha_norm(alpha_mse_values))
        bank_norm = Normalize(vmin=min(bank_mse_values), vmax=max(bank_mse_values))
        bank_colors = plt.cm.viridis(bank_norm(bank_mse_values))

        plt.figure(figsize=(18, 9))
        ax1 = plt.subplot(1, 2, 1, polar=True)
        scat1 = ax1.scatter(np.radians(X_df['azimuth']), X_df['range'], c=alpha_colors, alpha=0.75)
        ax1.set_title('Alpha Values Polar Plot')
        cbar1 = plt.colorbar(plt.cm.ScalarMappable(norm=alpha_norm, cmap='viridis'), ax=ax1)
        cbar1.set_label('Alpha MSE')
        cbar1.set_ticks([min(alpha_mse_values), max(alpha_mse_values)])

        ax2 = plt.subplot(1, 2, 2, polar=True)
        scat2 = ax2.scatter(np.radians(X_df['azimuth']), X_df['range'], c=bank_colors, alpha=0.75)
        ax2.set_title('Bank Values Polar Plot')
        cbar2 = plt.colorbar(plt.cm.ScalarMappable(norm=bank_norm, cmap='viridis'), ax=ax2)
        cbar2.set_label('Bank MSE')
        cbar2.set_ticks([min(bank_mse_values), max(bank_mse_values)])

        combined_mse_values = np.array(alpha_mse_values) + np.array(bank_mse_values)
        # Find the minimum combined MSE value
        min_combined_mse = min(combined_mse_values)
        
        # Find the index of the minimum combined MSE values
        min_combined_index = combined_mse_values.tolist().index(min_combined_mse)

        # Get the azimuth and range associated with the lowest combined MSE values
        min_combined_azimuth = azimuths[min_combined_index]
        min_combined_range = ranges[min_combined_index]
        min_alpha_mse = alpha_mse_values[min_combined_index]
        min_bank_mse = bank_mse_values[min_combined_index]

        self.lowest_azimuth = min_combined_azimuth
        self.lowest_range = min_combined_range

        # Output the minimum combined MSE values and associated azimuth and range
        print("The lowest combined Alpha and Bank MSE value is:", min_combined_mse)
        print("The associated azimuth for the lowest combined MSE is:", min_combined_azimuth)
        print("The associated range for the lowest combined MSE is:", min_combined_range)
        print("The Alpha MSE value at this point is:", min_alpha_mse)
        print("The Bank MSE value at this point is:", min_bank_mse)

        plt.suptitle(f"LAR MSE Heat Map for All Samples\nModel:{self.model.__class__.__name__}")
        plt.tight_layout()
        plt.savefig("lar_alpha_bank.png")

    ###### MSE Grid Plot #######   
    def plot_mse_grid(self, X_df, scaler_x, scaler_y):
        mse_values = []
        for index, row in X_df.iterrows():
            azimuth = row['azimuth']
            range_ = row['range']
            predicted_alpha, predicted_bank = self.predicting(azimuth, range_, scaler_x, scaler_y)

            actual_alpha = np.array(row['alpha_list'])[0]  
            actual_bank = np.array(row['bank_list'])[0]  
            if actual_alpha is None or actual_bank is None:
                print(f"Warning: Missing actual values for row index {index}.")
            alpha_mse = np.mean((predicted_alpha - actual_alpha) ** 2)
            bank_mse = np.mean((predicted_bank - actual_bank) ** 2)

            mse_values.append((alpha_mse, bank_mse)) 

        azimuths = np.array(X_df['azimuth'])  
        ranges = np.array(X_df['range']) 

        alpha_mse_values = [mse[0] for mse in mse_values]   
        bank_mse_values = [mse[1] for mse in mse_values]  

        alpha_mse_grid, azimuth_bins, range_bins = np.histogram2d(azimuths, ranges, bins=[30, 30], weights=alpha_mse_values)
        bank_mse_grid, _, _ = np.histogram2d(azimuths, ranges, bins=[30, 30], weights=bank_mse_values)

        # replace zero values with NaNs for masking
        alpha_mse_grid = np.where(alpha_mse_grid == 0, np.nan, alpha_mse_grid)
        bank_mse_grid = np.where(bank_mse_grid == 0, np.nan, bank_mse_grid)

        alpha_norm = Normalize(vmin=np.nanmin(alpha_mse_grid), vmax=np.nanmax(alpha_mse_grid))
        bank_norm = Normalize(vmin=np.nanmin(bank_mse_grid), vmax=np.nanmax(bank_mse_grid))

        plt.figure(figsize=(12, 6))

        # subplot for Alpha MSE
        ax1 = plt.subplot(1, 2, 1)
        c1 = ax1.imshow(alpha_norm(alpha_mse_grid), aspect='auto', cmap='viridis',
                        extent=[np.min(azimuths), np.max(azimuths), np.min(ranges), np.max(ranges)],
                        origin='lower')
        ax1.set_title('Alpha MSE Heatmap')
        ax1.set_xlabel('Azimuth (radians)')
        ax1.set_ylabel('Range')
        plt.colorbar(c1, ax=ax1, label='Alpha MSE')

        # subplot for Bank MSE
        ax2 = plt.subplot(1, 2, 2)
        c2 = ax2.imshow(bank_norm(bank_mse_grid), aspect='auto', cmap='viridis',
                        extent=[np.min(azimuths), np.max(azimuths), np.min(ranges), np.max(ranges)],
                        origin='lower')
        ax2.set_title('Bank MSE Heatmap')
        ax2.set_xlabel('Azimuth (radians)')
        ax2.set_ylabel('Range')
        plt.colorbar(c2, ax=ax2, label='Bank MSE')

        plt.suptitle(f"Grid MSE Heat Map for All Samples\nModel:{self.model.__class__.__name__}")
        plt.tight_layout()
        plt.savefig("grid_alpha_bank.png")  





    ###### actual vs predicted plots ######
    def plot_actual_vs_predicted(self, X_test, y_test, scaler_x, scaler_y):

        # Arrays to store combined actual and predicted values
        all_actual_alpha = []  # For actual alpha
        all_actual_bank = []   # For actual bank
        all_predicted_alpha = []  # For predicted alpha
        all_predicted_bank = []   # For predicted bank
        errors_alpha = []  # Absolute errors for alpha
        errors_bank = []   # Absolute errors for bank

        fig, axs = plt.subplots(4, 1, figsize=(14, 10))

        # Iterate through all test samples
        for index in range(len(X_test)):
            actual_y_values = y_test[index].cpu().numpy()
            actual_y_values = scaler_y.inverse_transform(actual_y_values).squeeze()
            actual_alpha, actual_bank = actual_y_values[0], actual_y_values[1]
            
            actual_x_values = X_test[index].cpu().numpy().reshape(1, -1)
            actual_x_values = scaler_x.inverse_transform(actual_x_values).squeeze()
            actual_azimuth, actual_range = actual_x_values[0], actual_x_values[1]

            predicted_alpha, predicted_bank = self.predicting(actual_x_values[0], actual_x_values[1], scaler_x, scaler_y)
            x_axis = np.arange(len(actual_alpha))
            axs[0].plot(x_axis, actual_alpha, label=f'({actual_azimuth:.2f}, {actual_range:.2f})', linestyle='-')
            axs[0].set_title("Actual Alpha")
            # axs[0].legend()

            axs[1].plot(x_axis, actual_bank, label=f'({actual_azimuth:.2f}, {actual_range:.2f})', linestyle='-')
            axs[1].set_title("Actual Bank")
            # axs[1].legend()

            axs[2].plot(x_axis, predicted_alpha, label=f'({actual_azimuth:.2f}, {actual_range:.2f})', linestyle='-')
            axs[2].set_title("Predicted Alpha")
            # axs[2].legend()

            axs[3].plot(x_axis, predicted_bank, label=f'({actual_azimuth:.2f}, {actual_range:.2f})', linestyle='-')
            axs[3].set_title("Predicted Bank")
            # axs[3].legend()
            # Track the predicted and actual values
            all_actual_alpha.append(actual_alpha.tolist())
            all_actual_bank.append(actual_bank.tolist())
            all_predicted_alpha.append(predicted_alpha.tolist())
            all_predicted_bank.append(predicted_bank.tolist())

            # Compute errors
            error_alpha = np.mean(abs((actual_alpha - predicted_alpha)**2))
            error_bank = np.mean(abs((actual_bank - predicted_bank)**2))
            errors_alpha.append(error_alpha)
            errors_bank.append(error_bank)

        # Add a global title and save the figure
        fig.suptitle(f'All Test Samples (10%) of Alpha and Bank\nModel:{self.model.__class__.__name__}', fontsize=16)
        plt.tight_layout()

        plt.savefig('actual_vs_predicted_all.png')  # Save as horizontal layout
        plt.close()

        print(np.array(all_actual_alpha).shape)
        print(np.array(errors_alpha).shape)

        # Find the indices of the minimum errors
        min_error_alpha_index = np.argmin(errors_alpha)
        min_error_bank_index = np.argmin(errors_bank)
        min_error_alpha = errors_alpha[min_error_alpha_index]
        min_error_bank = errors_alpha[min_error_bank_index]

        lowest_alpha_error_alpha_actual = all_actual_alpha[min_error_alpha_index]
        lowest_alpha_error_bank_actual = all_actual_bank[min_error_alpha_index]

        lowest_bank_error_alpha_actual = all_actual_alpha[min_error_bank_index]
        lowest_bank_error_bank_actual = all_actual_bank[min_error_bank_index]

        lowest_alpha_error_alpha_predicted = all_predicted_alpha[min_error_alpha_index]
        lowest_alpha_error_bank_predicted = all_predicted_bank[min_error_alpha_index]

        lowest_bank_error_alpha_predicted = all_predicted_alpha[min_error_bank_index]
        lowest_bank_error_bank_predicted = all_predicted_bank[min_error_bank_index]

        # Print results for the lowest errors
        fig, axs = plt.subplots(2, 1, figsize=(14, 10))

        # Plot Alpha
        axs[0].plot(x_axis, lowest_alpha_error_alpha_actual, label=f'Actual (Alpha Error {min_error_alpha:.4f})', color='blue')
        axs[0].plot(x_axis, lowest_alpha_error_alpha_predicted, label=f'Predicted (Alpha Error {min_error_alpha:.4f})', color='cyan')
        axs[0].plot(x_axis, lowest_bank_error_alpha_actual, label=f'Actual (Bank Error {min_error_bank:.4f})', color='orange')
        axs[0].plot(x_axis, lowest_bank_error_alpha_predicted, label=f'Predicted (Bank Error {min_error_bank:.4f})', color='red')
        axs[0].set_title("Alpha")
        axs[0].legend(loc='best')  # Automatically position the legend in the best location

        # Plot Bank
        axs[1].plot(x_axis, lowest_alpha_error_bank_actual, label=f'Actual (Alpha Error {min_error_alpha:.4f})', color='blue')
        axs[1].plot(x_axis, lowest_alpha_error_bank_predicted, label=f'Predicted (Alpha Error {min_error_alpha:.4f})', color='cyan')
        axs[1].plot(x_axis, lowest_bank_error_bank_actual, label=f'Actual (Bank Error {min_error_bank:.4f})', color='orange')
        axs[1].plot(x_axis, lowest_bank_error_bank_predicted, label=f'Predicted (Bank Error {min_error_bank:.4f})', color='red')
        axs[1].set_title("Bank")
        axs[1].legend(loc='best')  # Automatically position the legend in the best location

        fig.suptitle(f'Best of Alpha and Bank\nModel:{self.model.__class__.__name__}')
        plt.tight_layout()

        plt.savefig('actual_vs_predicted_best.png')

    def moving_average(self, data, window_size=5):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


class NN_Models():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)   

    def train_onetomany(self, batch_X, batch_y):
        batch_X = batch_X.unsqueeze(1)
        predictions = self.model(batch_X)
        return predictions
    
    def train_manytoone(self, start, window_size, seq_length, batch_X, batch_y, window_y=None):
        end = start + window_size

        if end > seq_length:
            end = seq_length

        if start == 0:
            # Initialize window_y to zeros for the first iteration
            window_y = torch.zeros(batch_y.size(0), batch_y.size(1), window_size).to(batch_y.device)
        else:
            # Ensure the shape of window_y for consistency
            if window_y is None or window_y.shape != (batch_y.size(0), batch_y.size(1), window_size):
                window_y = torch.zeros(batch_y.size(0), batch_y.size(1), window_size).to(batch_y.device)
        # print("window_y:", window_y.shape)

        # Expand batch_X to match window size
        window_X = batch_X.unsqueeze(2).expand(-1, -1, window_size)  # Shape: [batch_X.size(0), batch_X.size(1), window_size]
        # print("window_X:", window_X.shape)

        # Concatenate window_X and window_y along axis=1
        window_input = torch.cat((window_X, window_y), dim=1)  # Shape: [batch_X.size(0), 4, window_size]

        # Permute to match LSTM input expectations
        window_input = window_input.permute(0, 2, 1)  # Shape: [batch_X.size(0), window_size, 4]
        # print(f"window_input (start {start}): {window_input.shape}")

        # Model prediction
        prediction = self.model(window_input)  # Shape: [batch_X.size(0), 2]

        # Determine the target
        target = batch_y[:, :, start] if start < seq_length else batch_y[:, :, -1]  # Shape: [batch_X.size(0), 2]

        # Update window_y for the next iteration
        if start < seq_length:
            window_y = torch.cat((window_y[:, :, 1:], prediction.unsqueeze(2)), dim=2)  # Keep size [64, 2, 10]

        return prediction.unsqueeze(1), target.unsqueeze(1)
    
    def train_transformer(self, start, window_size, seq_length, batch_X, batch_y):
        end = start + window_size
        
        if end > seq_length:
            end = seq_length

        if start != 0:
            window_y = torch.cat((window_y[:, :, 1:], final_prediction.unsqueeze(2)), dim=2)

        window_X = batch_X.unsqueeze(2).expand(-1, -1, window_size)

        window_X = window_X[:, :, start:end]  
        window_y_target = batch_y[:, :, start:end] 

        predictions = self.model(window_X)
        final_prediction = predictions[:, -1, :].detach()
        return predictions, window_y_target
    





"""
Created on Thu Jun 26 2025

@author: imoore
"""

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


from sklearn import svm, datasets

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import make_scorer

import multiprocessing
import itertools
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer
from functools import partial
import joblib
from torch.utils.data import DataLoader, TensorDataset
import math

from nn_models import FlexibleNeuralNet, LSTMModel_OnetoMany, LSTMModel_ManytoOne, TransformerSeq2Seq, ModelWrapper
from nn_functions import NN_Functions, NN_Models

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MASTER_ADDR = os.environ["MASTER_ADDR"]
MASTER_PORT = os.environ["MASTER_PORT"]


torch.distributed.init_process_group(
    backend="nccl", world_size=WORLD_SIZE, rank=RANK
)

# Loading db dataset
def load_results(db_filename):
    connection = sqlite3.connect(db_filename)
    results_df = pd.read_sql_query('SELECT * FROM results', connection)
    boost11_df = pd.read_sql_query('SELECT * FROM boost_11_timeseries', connection)
    terminal_df = pd.read_sql_query('SELECT * FROM terminal_timeseries', connection)
    
    # Filter successful runs
    successful_runs_df = results_df[results_df['status'] == 1]
    successful_boost_df = boost11_df[boost11_df['case_num'].isin(successful_runs_df['case_num'])]
    successful_terminal_df = terminal_df[terminal_df['case_num'].isin(successful_runs_df['case_num'])]
    
    connection.close()
    
    return successful_runs_df, successful_boost_df, successful_terminal_df

# Formatting df 
def prepare_df(db_filename):
    results_df, boost11_df, terminal_df = load_results(db_filename)
    
    # Combine and filter data
    merged_df = pd.concat([boost11_df, terminal_df], ignore_index=True)

    # Group by case_num and aggregate alpha and bank into lists
    aggregated_df = merged_df.groupby('case_num').agg(
        alpha_list=('alpha', lambda x: list(x)),
        bank_list=('bank', lambda x: list(x))
    ).reset_index()


    X_train_df = pd.merge(results_df, aggregated_df, on='case_num')


    return X_train_df
def moving_average(values, window_size):
    """Calculate the moving average of a 2D array."""
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

# Data preparation function
def prepare_data(db_filename, save_new_dataset):
    X_train_df = prepare_df(db_filename)

    print(X_train_df['status'])

    X = torch.stack((
        torch.tensor(X_train_df['azimuth'], dtype=torch.float32),
        torch.tensor(X_train_df['range'], dtype=torch.float32)
    ), dim=1)


    y = torch.stack((
        torch.tensor(X_train_df['alpha_list'], dtype=torch.float32),
        torch.tensor(X_train_df['bank_list'], dtype=torch.float32)
    ), dim=1)

    smoothing_window=15

    # Apply moving average smoothing to each time series in the targets
    num_samples, num_targets, num_timesteps = y.shape
    smoothed_length = num_timesteps - smoothing_window + 1
    y_smoothed = np.zeros((num_samples, num_targets, smoothed_length))

    for i in range(num_targets):
        for j in range(num_samples): 
            y_smoothed[j, i, :] = moving_average(y[j, i], window_size=smoothing_window)
    
    # Convert back to torch tensor after smoothing
    y = torch.tensor(y_smoothed, dtype=torch.float32)
    
    # Adjust the size of X to match the reduced size of y after smoothing
    if smoothing_window > 1:
        adjustment = smoothing_window - 1
        X = X[adjustment:, :]

    num_cases = len(X)
    shuffled_indices = np.random.permutation(num_cases)
    split_index = int(0.9 * num_cases)
    
    train_indices = shuffled_indices[:split_index]
    test_indices = shuffled_indices[split_index:]

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)  
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape) # 3D shaping

    X_train = X_scaled[train_indices]
    y_train = y_scaled[train_indices]
    X_test = X_scaled[test_indices]
    y_test = y_scaled[test_indices]


    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

    if save_new_dataset:
        data_to_save = (X_train, X_test, y_train, y_test, scaler_x, scaler_y)
        with open('data.pkl', 'wb') as f:
            pickle.dump(data_to_save, f)

    return X_train, X_test, y_train, y_test, scaler_x, scaler_y

def generate_square_subsequent_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

def train_model(X_train, X_test, y_train, y_test, model, model_type, num_epochs, batch_size, patience, l1_lambda, learning_rate, window_size, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    starttime = time.time()
    train_losses, test_losses = [], []
    best_test_loss = float('inf')  
    patience_counter = 0
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    seq_length = y_train.size(2)  
  
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        nnm = NN_Models(model, device)
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            if model_type == 'flex' or model_type == 'lstm_onetomany':
                optimizer.zero_grad()
                predictions = nnm.train_onetomany(batch_X, batch_y)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                    
            if model_type == 'lstm_manytoone':
                # Loop to handle full sequence predictions
                predictions = []
                targets = []

                for start in range(0, seq_length + 1):
                    prediction, target = nnm.train_manytoone(start, window_size, seq_length, batch_X, batch_y)
                    
                    predictions.append(prediction)  
                    targets.append(target)  
                    
                # Concatenate along sequence dimension
                predictions = torch.cat(predictions, dim=1)  
                targets = torch.cat(targets, dim=1)

                loss = criterion(predictions, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()  

            if model_type == 'transformer':
                window_y = torch.zeros(batch_size, batch_y.size(2), window_size).to(device) 

                for start in range(0, seq_length, window_size):  
                    predictions, window_y_target = nnm.train_transformer(start, window_size, seq_length, batch_X, batch_y)
                    loss = criterion(predictions, window_y_target)

                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    l1_loss = l1_lambda * l1_norm
                    loss += l1_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()   

        train_loss /= len(data_loader)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        model.eval()
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            test_loss = 0
            window_y = torch.zeros(batch_size, batch_y.size(2), window_size).to(device) 

            for batch_X, batch_y in test_loader:

                if model_type == 'flex' or model_type == 'lstm_onetomany':
                    batch_X_test, batch_y_test = batch_X.to(device), batch_y.to(device)
                    predictions = nnm.train_onetomany(batch_X_test, batch_y_test)
                    loss = criterion(predictions, batch_y)

                    test_loss += loss.item()
                    
                if model_type == 'lstm_manytoone':

                    predictions = []
                    targets = []

                    for start in range(0, seq_length + 1):
                        prediction, target = nnm.train_manytoone(start, window_size, seq_length, batch_X, batch_y)
                        
                        predictions.append(prediction)  
                        targets.append(target)  
                        
                    # Concatenate along sequence dimension
                    predictions = torch.cat(predictions, dim=1)  
                    targets = torch.cat(targets, dim=1)

                    loss = criterion(predictions, targets)
                    test_loss += loss.item()  

                if model_type == 'transformer':

                    # Loop to handle full sequence predictions
                    predictions = []
                    targets = []

                    for start in range(0, seq_length + 1):
                        prediction, target = nnm.train_manytoone(start, window_size, seq_length, batch_X, batch_y)
                        
                        predictions.append(prediction)  
                        targets.append(target)  
                        
                    # Concatenate along sequence dimension
                    predictions = torch.cat(predictions, dim=1)  
                    targets = torch.cat(targets, dim=1)

                    loss = criterion(predictions, targets)

                    test_loss += loss.item()  
            test_loss /= len(test_loader)
            test_losses.append(test_loss)            
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        if (epoch + 1) % 1 == 0:
            elapsed_time = time.time() - starttime
            print(f"Epoch {epoch + 1}/{num_epochs}: Train loss: {train_loss:.5E}, Test loss: {test_loss:.5E}, Time elapsed: {elapsed_time:.2f} sec")

    return model, train_losses, test_losses

def custom_scorer(y_true, y_pred):
    y_pred_reshaped = y_pred.reshape(y_true.shape)
    return -np.mean((y_pred_reshaped - y_true) ** 2)

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def apply_smoothing(tensor, window_size):
    smoothed_tensors = []
    for i in range(tensor.size(0)):
        if tensor.dim() == 2:
            smoothed_tensor = moving_average(tensor[i].cpu().numpy(), window_size)
        elif tensor.dim() == 3:
            smoothed_tensor = np.array([moving_average(tensor[i, j].cpu().numpy(), window_size) for j in range(tensor.size(1))])
        else:
            raise ValueError("Unsupported tensor dimension for smoothing.")
        smoothed_tensors.append(smoothed_tensor)
    
    return torch.tensor(smoothed_tensors, dtype=torch.float32).to(device)

if __name__ == "__main__":
    start_time = time.time()

    ###### Set Argument Parser ######
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-e", "--epochs", required=True, type=int, help="number of epochs")
    
    # args = parser.parse_args()
    # num_epochs = args.epochs

    ###### Set Parameters ######
    model_type = 'lstm_manytoone'
    # flex,  lstm_onetomany, lstm_manytoone, transformer

    rerun_training = True
    save_new_model = True
 
    recreate_dataset = True # reshuffle?
    save_new_dataset = True

    batch_size = 64
    hidden_size = 32  
    num_layers = 3
    patience = 500
    l1_lambda = 0.0001
    learning_rate = 0.001
    num_epochs = 10000
    dropout = 0.5
    window_size=10


    # Load generated dataset
    db_filename = '/home/imoore/misslemdao/trajectory_results_azimuth21_range51_cores30_date_2025_06_25_time_2035.db'

    if recreate_dataset:
        X_train, X_test, y_train, y_test, scaler_x, scaler_y = prepare_data(db_filename, save_new_dataset)
    else:
        with open('data.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test, scaler_x, scaler_y = pickle.load(f)

    device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

    torch.cuda.set_device(device)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    if model_type == 'flex':
        input_size = 2  
        output_size = 212 
        output_shape = (2, 106)

        activation_type = "ReLU"

        layer_sizes = [input_size] + [hidden_size] * num_layers + [output_size]  
        print(f"Layer Sizes: {layer_sizes}")
        
        model = FlexibleNeuralNet(layer_sizes=layer_sizes, activation_type=activation_type, dropout=dropout).to(device)

    if model_type == 'lstm_onetomany':
        input_dim = 2
        output_dim = 2
        
        model = LSTMModel_OnetoMany(input_dim, output_dim, hidden_size, num_layers, dropout).to(device)

    if model_type == 'lstm_manytoone':
        input_dim = 4
        output_dim = 2
        patience = 5
        
        model = LSTMModel_ManytoOne(input_dim, output_dim, hidden_size, num_layers, dropout).to(device)

    if model_type == 'transformer':
        input_dim = 2
        output_dim = 2
        seq_length = 106  

        activation_type = "ReLU"

        model = TransformerSeq2Seq(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            seq_length=seq_length
        ).to(device)

    print(model)
    """
    Grid Search for Hyperparameter Tuning
    """
    # X_train_numpy = X_train.cpu().numpy()
    # y_train_numpy = y_train.cpu().numpy()
    # print("X_train_numpy.shape:", X_train_numpy.shape)
    # print("y_train_numpy.shape:", y_train_numpy.shape)

    # param_grid = {
    #     'hidden_size': [32, 64, 128],
    #     'num_layers': [1, 2, 3, 4], 
    #     # 'activation_type': ["tanh", "ReLU"],
    #     'epochs': [10000],
    #     'lr': [0, 0.01, 0.001, 0.0001], 
    #     'batch_size': [32, 64, 128, 256],
    #     'patience': [300],
    #     'l1_lambda': [0, 0.0001, 0.001],
    #     'dropout': [0.1, 0.25, 0.5]
    # }

    # model_grid_search = ModelWrapper(model_type="lstm_onetomany")
    # scorer = make_scorer(custom_scorer, greater_is_better=True)

    # with joblib.parallel_backend('threading'):
    #     grid_search = GridSearchCV(
    #         estimator=model_grid_search,   # model wrapper object
    #         param_grid=param_grid,         # hyperparameter combinations
    #         scoring=scorer,                # custom scoring function
    #         n_jobs=2,                      # maximum parallel jobs
    #         cv=3                           # cross-validation splits
    #     )        
    #     grid_search.fit(X_train_numpy, y_train_numpy)
    # best_model = grid_search.best_estimator_
    # print(f"Best Parameters: {grid_search.best_params_}")
    # print(f"Best Score: {grid_search.best_score_}")

    if rerun_training: 
        model, train_losses, test_losses = train_model(
            X_train, X_test, y_train, y_test, 
            model,
            model_type=model_type,
            num_epochs=num_epochs, 
            device=device,
            batch_size=batch_size,
            patience=patience,
            l1_lambda=l1_lambda,
            learning_rate=learning_rate,
            window_size=window_size
        )
        torch.distributed.destroy_process_group()

    else:
        torch.distributed.destroy_process_group()
        # model.load_state_dict(torch.load('trajectory_nn_model_2.pth'))
        
    if save_new_model:
        model_state = {
            'model_state_dict': model.state_dict(),
            'X_train': X_train, 
            'y_train': y_train, 
            'X_test': X_test, 
            'y_test': y_test, 
            'scaler_x': scaler_x,
            'scaler_y': scaler_y
        }

        torch.save(model_state, 'trajectory_nn_model_2.pth')
    model.eval()  

    fn = NN_Functions(model, device)

    # fn.plot_data_sections(X_test, y_test, scaler_x, scaler_y)

    if rerun_training:
        fn.plot_loss_vs_epoch(train_losses, test_losses)
    

    X_df = prepare_df(db_filename)

    fn.plot_mse_polar(X_df, scaler_x, scaler_y)

    fn.plot_mse_grid(X_df, scaler_x, scaler_y)

    fn.plot_actual_vs_predicted(X_test, y_test, scaler_x, scaler_y)


