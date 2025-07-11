
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




RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MASTER_ADDR = os.environ["MASTER_ADDR"]
MASTER_PORT = os.environ["MASTER_PORT"]


torch.distributed.init_process_group(
    backend="nccl", world_size=WORLD_SIZE, rank=RANK
)

# Load results function
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


# Data preparation function
def prepare_data(db_filename):
    X_train_df = prepare_df(db_filename)

    # print(X_train_df)
    # print(X_train_df.columns)

    # Stack features: azimuth and range, both should have length 240
    X = torch.stack((
        torch.tensor(X_train_df['azimuth'], dtype=torch.float32),
        torch.tensor(X_train_df['range'], dtype=torch.float32)
    ), dim=1)


    y = torch.stack((
        torch.tensor(X_train_df['alpha_list'], dtype=torch.float32),
        torch.tensor(X_train_df['bank_list'], dtype=torch.float32)
    ), dim=1)

    num_cases = len(results_df['case_num'])
    shuffled_indices = np.random.permutation(num_cases)
    split_index = int(0.9 * num_cases)
    
    train_indices = shuffled_indices[:split_index]
    test_indices = shuffled_indices[split_index:]

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)  
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape) # 3D shaping

    # Create scaled training and testing sets
    X_train = X_scaled[train_indices]
    y_train = y_scaled[train_indices]
    X_test = X_scaled[test_indices]
    y_test = y_scaled[test_indices]


    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')


    # Save the datasets and scalers
    data_to_save = (X_train, X_test, y_train, y_test, scaler_x, scaler_y)
    with open('data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    return X_train, X_test, y_train, y_test, scaler_x, scaler_y


def train_model(X_train, X_test, y_train, y_test, model,
                input_size, output_size, num_epochs, device, batch_size=256, patience=500, l1_lambda=0.01):
    criterion = nn.MSELoss()
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    starttime = time.time()
    train_losses, test_losses = [], []

    best_test_loss = float('inf')  
    epochs_without_improvement = 0  

    # Initialize hidden and memory states
    hidden_size = 128   # Should match the LSTM configuration
    num_layers = 1      # Number of LSTM layers
    # Initialize to zeros
    hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device) 
    memory = torch.zeros(num_layers, batch_size, hidden_size).to(device)
    for epoch in range(num_epochs):
        try:
            indices = torch.randperm(X_train.size(0)).to(device)

            # Training phase
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X_train[batch_indices]  
                batch_y = y_train[batch_indices] 
    
                outputs, hidden, memory = model(batch_X, hidden, memory)                
                # outputs = outputs.view(-1, 2, 120) 
                # train_acc += (outputs.argmax(1) == labels).sum().item()
                training_loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                training_loss.backward()
                optimizer.step()
                    
            # Evaluation phase
            model.eval() 
            with torch.no_grad():
                test_outputs = model(X_test).view(-1, 2, 120)
                test_loss = criterion(test_outputs, y_test)

            time_elapsed = (time.time() - starttime) / 60
            train_losses.append(training_loss.cpu().item())
            test_losses.append(test_loss.cpu().item())   

            # Early stopping logic
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                epochs_without_improvement = 0  
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered. Stopping training at epoch {epoch + 1}.")
                break

            if (epoch + 1) % 100 == 0: 
                logstring = "Epoch %d: time elapsed %.2f, train loss %.3E, test loss %.3E" % (
                    epoch + 1, time_elapsed, training_loss.item(), test_loss.item()
                )

                print(logstring)

        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT \n")
            torch.distributed.destroy_process_group()
            break

    return model, train_losses, test_losses

class FlexibleNeuralNet(nn.Module):
    def __init__(self, layer_sizes, activation_type="tanh"):

        super(FlexibleNeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []

        # Create N-1 layers with an activation func
        for i in range(len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # Optionally add Tanh activation function here or any other activation
            if i != 0:
                self.layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            if activation_type == "tanh":
                self.layers.append(nn.Tanh())
            elif activation_type == "ReLU":
                self.layers.append(nn.LeakyReLU())
            elif activation_type == "SiLU":
                self.layers.append(nn.SiLU())
            else:
                pass
            # add drop out layers but not after input layer
            if i != 0:
                self.layers.append(nn.Dropout(p=0.5))

        # add ouput layer
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            self.activations.append(x)  
        return x

class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlockMLP, self).__init__()
        self.norm1 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.norm2 = nn.LayerNorm(input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, output_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.act(self.norm1(x))
        skip = self.fc3(x)
        x = self.act(self.norm2(self.fc1(x)))
        x = self.fc2(x)
        return x + skip

class LSTM(nn.Module):
    def __init__(self, seq_len, num_features, output_size, num_blocks=1):
        super(LSTM, self).__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(seq_len * num_features, 4 * seq_len * num_features),  # Corrected initialization
            nn.ELU(),
            nn.Linear(4 * seq_len * num_features, 128)  # Output size for MLP
        )
        
        # Create LSTM block
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        
        # Residual blocks
        blocks = [ResBlockMLP(128, 128) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        # Output layer
        self.fc_out = nn.Linear(128, output_size)

    def forward(self, input_seq, hidden_in, mem_in):
        # Reshape the input sequence for the MLP
        input_vec = self.input_mlp(input_seq.view(input_seq.size(0), -1))  # Reshape to [batch_size, seq_len*num_features]
        input_vec = input_vec.unsqueeze(1)  # Shape: [batch_size, 1, 128] for LSTM

        # Pass through LSTM
        output, (hidden_out, mem_out) = self.lstm(input_vec, (hidden_in, mem_in))

        # Process last time step output through residual blocks
        x = output[:, -1, :]  # Taking the last time step output
        x = self.res_blocks(x)  # Residual blocks

        # Output fully connected layer
        return self.fc_out(x), hidden_out, mem_out
if __name__ == "__main__":
    #### Set up argument parser ####
    # # parser = argparse.ArgumentParser()
    # # parser.add_argument("-e", "--epochs", required=True, type=int, help="number of epochs")
    
    # args = parser.parse_args()
    # num_epochs = args.epochs
    
    db_filename = '/home/imoore/misslemdao/trajectory_results_azimuth21_range51_cores30_date_2025_06_25_time_2035.db'

    # X_train, X_test, y_train, y_test, scaler_x, scaler_y = prepare_data(db_filename)
    with open('data.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test, scaler_x, scaler_y = pickle.load(f)

    device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)  
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device) 
    
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)  
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device) 

    input_size = X_train.shape[1]
    output_size = y_train.shape

    layer_sizes = [2, 32, 64, 64, 240]  

    # model = FlexibleNeuralNet(layer_sizes=layer_sizes, activation_type='ReLU').to(device)
    size_timeseries = 120
    seq_len = 120  # Length of the time series (number of time steps)
    num_features = 2  # Number of input features (parameters)
    print("OUtput Size:", output_size)
    output_size = y_train.shape[1]  # Number of outputs required (240 based on your original requirement)

    # Create the model
    model = LSTM(seq_len=seq_len, num_features=num_features, output_size=output_size, num_blocks=2).to(device)
    print(model)  # Print model architecture    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False)

    model, train_losses, test_losses = train_model(
        X_train, X_test, y_train, y_test, 
        model,
        input_size=input_size, 
        output_size=output_size, 
        num_epochs=15_000, 
        device=device
    )
    
    # torch.save(model.state_dict(), 'trajectory_nn_model.pth')
    # torch.distributed.destroy_process_group()
    # model.load_state_dict(torch.load('trajectory_nn_model.pth'))
    # torch.distributed.destroy_process_group()

    model.eval()  


    #### mse heat map global
    X_df = prepare_df(db_filename)
    mse_values = []
    for index, row in X_df.iterrows():
        azimuth = row['azimuth']
        range_ = row['range']
        input_tensor = np.array([[azimuth, range_]], dtype=np.float32)

        input_tensor_scaled = scaler_x.transform(input_tensor)

        input_tensor_scaled = torch.tensor(input_tensor_scaled, dtype=torch.float32).to(device)
        input_tensor_scaled = input_tensor_scaled.squeeze(1)  
        
        with torch.no_grad():
            output = model(input_tensor_scaled)
            predicted_values = output.view(-1, 2, 120)  
            predicted_values = predicted_values.squeeze(0).cpu()  
            predicted_values = scaler_y.inverse_transform(predicted_values.numpy())  

        predicted_alpha = predicted_values[0]  
        predicted_bank = predicted_values[1] 

        actual_alpha = np.array(row['alpha_list'])[0]  
        actual_bank = np.array(row['bank_list'])[0]  

        alpha_mse = np.mean((predicted_alpha - actual_alpha) ** 2)
        bank_mse = np.mean((predicted_bank - actual_bank) ** 2)

        mse_values.append((alpha_mse, bank_mse)) 

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


    plt.tight_layout()
    plt.savefig("lar_alpha_bank.png")  


    #### grid plotting ###
    alpha_mse_grid, azimuth_bins, range_bins = np.histogram2d(azimuths, ranges, bins=[30, 30], weights=alpha_mse_values)
    bank_mse_grid, _, _ = np.histogram2d(azimuths, ranges, bins=[30, 30], weights=bank_mse_values)

    # Replace zero values with NaNs for masking
    alpha_mse_grid = np.where(alpha_mse_grid == 0, np.nan, alpha_mse_grid)
    bank_mse_grid = np.where(bank_mse_grid == 0, np.nan, bank_mse_grid)

    # Normalize the grids for proper color mapping (skip NaNs for min/max)
    alpha_norm = Normalize(vmin=np.nanmin(alpha_mse_grid), vmax=np.nanmax(alpha_mse_grid))
    bank_norm = Normalize(vmin=np.nanmin(bank_mse_grid), vmax=np.nanmax(bank_mse_grid))

    # Create pixel plots using imshow
    plt.figure(figsize=(12, 6))

    # First subplot for Alpha MSE
    ax1 = plt.subplot(1, 2, 1)
    c1 = ax1.imshow(alpha_norm(alpha_mse_grid), aspect='auto', cmap='viridis',
                    extent=[np.min(azimuths), np.max(azimuths), np.min(ranges), np.max(ranges)],
                    origin='lower')
    ax1.set_title('Alpha MSE Heatmap')
    ax1.set_xlabel('Azimuth (radians)')
    ax1.set_ylabel('Range')
    plt.colorbar(c1, ax=ax1, label='Alpha MSE')

    # Second subplot for Bank MSE
    ax2 = plt.subplot(1, 2, 2)
    c2 = ax2.imshow(bank_norm(bank_mse_grid), aspect='auto', cmap='viridis',
                    extent=[np.min(azimuths), np.max(azimuths), np.min(ranges), np.max(ranges)],
                    origin='lower')
    ax2.set_title('Bank MSE Heatmap')
    ax2.set_xlabel('Azimuth (radians)')
    ax2.set_ylabel('Range')
    plt.colorbar(c2, ax=ax2, label='Bank MSE')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig("grid_alpha_bank.png")  



    # #### Loss Plot ####
    # plt.figure()
    # plt.plot(train_losses, label="Training Loss")
    # plt.plot(test_losses, label="Testing Loss")
    # plt.title("Loss over Epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)

    # plt.savefig('loss_plot.png')  
    # plt.close()  

    ####  activation plotting
    hidden_layer_indices = range(1, len(layer_sizes)-1) 
    print(hidden_layer_indices)

    plt.figure(figsize=(12, 12))  

    for idx, i in enumerate(hidden_layer_indices):
        activation = model.activations[i]  
        ax = plt.subplot(len(hidden_layer_indices), 1, idx + 1)  
        ax.imshow(activation.detach().cpu().numpy(), aspect='auto', cmap='hot')
        ax.set_title(f'Activations of Hidden Layer {i} ({layer_sizes[i]} Features)')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Sample Index')
        plt.colorbar(ax.imshow(activation.detach().cpu().numpy(), aspect='auto', cmap='hot'), ax=ax, label='Activation Value')

    plt.tight_layout()

    plt.savefig('activations_hidden_layers.png')
    plt.close()


    exit()

    #### actual vs predicted plots 
    random_index = 6
    random_azimuth = X_test[random_index][0]
    random_range = X_test[random_index][1]

    actual_values = y_test[random_index].cpu().numpy()
    actual_values = scaler_y.inverse_transform(actual_values)

    selected_values_scaled = np.array([[random_azimuth.cpu().numpy(), random_range.cpu().numpy()]]) 
    selected_values = scaler_x.inverse_transform(selected_values_scaled)
    input_tensor = torch.tensor([[random_azimuth, random_range]], dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)

        predicted_values = output.view(-1, 2, 120) 
        predicted_values = predicted_values.squeeze(0) 
        
        predicted_values = predicted_values.cpu()  
        predicted_values = scaler_y.inverse_transform(predicted_values.numpy())  

    x_axis = np.arange(120)

    # In the training loop (after model predictions):

    # Create a figure and an array of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column of subplots

    # First subplot for the first output
    axs[0].plot(x_axis, actual_values[0], label='Actual', linestyle='-')
    axs[0].plot(x_axis, predicted_values[0], label='Predicted', linestyle='-')

    axs[0].set_title('Alpha')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot for the second output
    axs[1].plot(x_axis, actual_values[1], label='Actual', linestyle='-')
    axs[1].plot(x_axis, predicted_values[1], label='Predicted', linestyle='-')
    axs[1].set_title('Bank')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout to prevent overlap
    fig.suptitle(f'Comparison of Actual and Predicted Values\n(azimuth: {selected_values[0][0]}, range: {selected_values[0][1]}, index: {random_index})')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_2.png')
    plt.close()  

    exit()


    mse_values = []
    azimuth_values = []
    range_values = []
    model.eval()

    with torch.no_grad():
        for index in range(X_test.shape[0]):
            input_tensor = torch.tensor([[X_test[index][0], X_test[index][1]]], dtype=torch.float32).to(device)

            output = model(input_tensor) 
            
            predicted_values = output.view(-1, 2, 120)  
            predicted_values = predicted_values.squeeze(0)  
            
            predicted_values = predicted_values.cpu().numpy()  
            predicted_values = scaler_y.inverse_transform(predicted_values) 
                
            actual_values = y_test[index].cpu().numpy()
            actual_values = scaler_y.inverse_transform(actual_values)
            mse = np.mean((predicted_values - actual_values) ** 2)

            azimuth_value = X_test[index][0]  # This is a tensor
            range_value = X_test[index][1]     # This is also a tensor
            
            # Create a 2D array for inverse transformation
            values_to_transform = np.array([[azimuth_value.cpu().item(), range_value.cpu().item()]])
            
            # Perform inverse transformation for both azimuth and range
            transformed_values = scaler_x.inverse_transform(values_to_transform)  # Should work now
            azimuth_transformed = transformed_values[0][0]  # Get transformed azimuth
            range_transformed = transformed_values[0][1]    # Get transformed range
            
            # Store the transformed values
            azimuth_values.append(azimuth_transformed)
            range_values.append(range_transformed)
            mse_values.append(mse)

    average_mse = np.mean(mse_values)

    plt.figure(figsize=(12, 6))
    plt.plot(mse_values, label='MSE per Sample', linestyle='-', marker='o', markersize=3)
    plt.axhline(y=average_mse, color='r', linestyle='--', label=f'Average MSE: {average_mse:.4f}')
    plt.title('Mean Squared Error Across Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)

    plt.savefig('mse_compare.png')
    plt.close()  



    # Convert lists to numpy arrays if they're not already
    azimuth_values = np.array(azimuth_values)  # Shape should be (n,)
    range_values = np.array(range_values)  # Shape should be (n,)
    mse_values = np.array(mse_values)  # Shape should be (n,)

    # Now create a meshgrid for plotting
    unique_azimuths = np.unique(azimuth_values)
    unique_ranges = np.unique(range_values)

    # Create a meshgrid
    X_range, Y_azimuth = np.meshgrid(unique_ranges, unique_azimuths)

    # Create a 2D array for the MSE values based on the mesh
    Z_mse = np.zeros((len(unique_azimuths), len(unique_ranges)))

    # Fill Z_mse with the corresponding MSE values
    for r in range(len(unique_ranges)):
        for a in range(len(unique_azimuths)):
            # Find indices for the original values in the flattened arrays
            mask = (range_values == unique_ranges[r]) & (azimuth_values == unique_azimuths[a])
            if np.any(mask):  # If there are corresponding MSE values
                Z_mse[a, r] = mse_values[mask][0]  # Get the first matching mse

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the surface
    surf = ax.plot_surface(X_range, Y_azimuth, Z_mse, cmap='viridis', edgecolor='none')

    # Customizing the plot
    ax.set_title('MSE as a Function of Range and Azimuth')
    ax.set_xlabel('Range')
    ax.set_ylabel('Azimuth')
    ax.set_zlabel('Mean Squared Error')
    fig.colorbar(surf, label='Mean Squared Error')  # Add a color bar for clarity
    # Save the plot to a file
    plt.show()

    plt.savefig('grad_global.png')

    # plt.close()  # Close the figure after saving
