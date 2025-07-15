
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
def prepare_data(db_filename, save_new_dataset):
    X_train_df = prepare_df(db_filename)

    X = torch.stack((
        torch.tensor(X_train_df['azimuth'], dtype=torch.float32),
        torch.tensor(X_train_df['range'], dtype=torch.float32)
    ), dim=1)


    y = torch.stack((
        torch.tensor(X_train_df['alpha_list'], dtype=torch.float32),
        torch.tensor(X_train_df['bank_list'], dtype=torch.float32)
    ), dim=1)

    num_cases = len(X_train_df['case_num'])
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


def train_model(X_train, X_test, y_train, y_test, model,
                input_size, output_size, num_epochs, device, batch_size, patience, l1_lambda, learning_rate):
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    starttime = time.time()
    train_losses, test_losses = [], []

    best_test_loss = float('inf')  
    epochs_without_improvement = 0  

    for epoch in range(num_epochs):
        model.train()

        try:
            indices = torch.randperm(X_train.size(0)).to(device)

            # Training phase
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X_train[batch_indices]  
                batch_y = y_train[batch_indices] 
                
                outputs = model(batch_X)
                outputs = outputs.view(-1, 2, 120)  # reshape to (batch_size, 2, 120)
                # print("Training Output Shape:", outputs.shape)
    
                # train_acc += (outputs.argmax(1) == labels).sum().item()
                training_loss = criterion(outputs, batch_y)
                # # Calculate L1 penalty
                # l1_norm = sum(p.abs().sum() for p in model.parameters())
                # l1_loss = l1_lambda * l1_norm
                
                # # Total loss
                # training_loss += l1_loss
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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(p=dropout_rate) 
        self.fc = nn.Linear(hidden_size, output_size)
        self.activations = []

    def forward(self, x):
        # print("LSTM input shape:", x.shape)  # Debug: Check input shape

        lstm_out, _ = self.lstm(x)  
        # print("LSTM output shape:", lstm_out.shape)  # Debug: Check LSTM output shape
        self.activations.append(lstm_out.detach().cpu())  

        if lstm_out.dim() == 3:  
            last_time_step = lstm_out[:, -1, :]  
        elif lstm_out.dim() == 2:  
            last_time_step = lstm_out  
        else:
            raise ValueError("Unexpected output shape from LSTM")
        
        last_time_step = self.dropout(last_time_step)  
        output = self.fc(last_time_step)  
        return output


def predicting(azimuth, range_, scaler_x, scaler_y):
    try:
        input_tensor = np.array([[azimuth, range_]], dtype=np.float32)
    except:
        input_tensor = torch.tensor([[azimuth, range_]], dtype=torch.float32).cpu().numpy()

    input_tensor_scaled = scaler_x.transform(input_tensor)

    input_tensor_scaled = torch.tensor(input_tensor_scaled, dtype=torch.float32).to(device)
    input_tensor_scaled = input_tensor_scaled.squeeze(1)  
    
    with torch.no_grad():
        output = model(input_tensor_scaled)
        predicted_values = output.view(-1, 2, 120)  
        predicted_values = predicted_values.squeeze(0).cpu()  
        predicted_values = scaler_y.inverse_transform(predicted_values.numpy())  

    return predicted_values[0], predicted_values[1], 

if __name__ == "__main__":

    ###### Set Argument Parser ######
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-e", "--epochs", required=True, type=int, help="number of epochs")
    
    # args = parser.parse_args()
    # num_epochs = args.epochs
    
    ###### Set Parameters ######
    input_size = 2  
    hidden_size = 64  
    num_layers = 3    
    output_size = 240  


    batch_size = 32
    patience = 500
    l1_lambda = 0.001
    learning_rate = 1e-5
    num_epochs = 300

    model_type = 'lstm' 

    rerun_training = True
    save_new_model = True
 
    recreate_dataset = False # reshuffle?
    save_new_dataset = False

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


    print("X_Train Input Shape:", X_train.shape)
    print("y_train Input Shape:", y_train.shape)

    if model_type == 'flex':
        layer_sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        print(f"Layer Sizes: {layer_sizes}")
        model = FlexibleNeuralNet(layer_sizes=layer_sizes, activation_type='ReLU').to(device)
    if model_type == 'lstm':
        model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if rerun_training: 
        model, train_losses, test_losses = train_model(
            X_train, X_test, y_train, y_test, 
            model,
            input_size=input_size, 
            output_size=output_size, 
            num_epochs=num_epochs, 
            device=device,
            batch_size=batch_size,
            patience=patience,
            l1_lambda=l1_lambda,
            learning_rate=learning_rate
        )
        torch.distributed.destroy_process_group()

    else:
        torch.distributed.destroy_process_group()
        model.load_state_dict(torch.load('trajectory_nn_model_2.pth'))
        
    if save_new_model:
        torch.save(model.state_dict(), f'trajectory_nn_model_{model_type}_hs{hidden_size}_bs{batch_size}_l1{l1_lambda}_lr{learning_rate}.pth') # _date_{datetime.now().strftime("%Y_%m_%d")}_time_{datetime.now().strftime("%H%M")}

    model.eval()  


    '''
    Plotting for Tuning
    '''

    ###### Loss vs Epochs Plot ######
    if rerun_training:
        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(test_losses, label="Testing Loss")
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig('loss_plot.png')  
        plt.close()  

    ###### MSE Polar Plot #######
    X_df = prepare_df(db_filename)
    mse_values = []
    for index, row in X_df.iterrows():
        azimuth = row['azimuth']
        range_ = row['range']
        predicted_alpha, predicted_bank = predicting(azimuth, range_, scaler_x, scaler_y)

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


    ###### MSE Grid Plot #######    
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

    plt.tight_layout()
    plt.savefig("grid_alpha_bank.png")  


    ######  Activation Plot ######
    if model_type == 'flex':
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

    if model_type == 'lstm':
        '''
        Activation References: https://www.mathworks.com/help/deeplearning/ug/visualize-features-of-lstm-network.html
        '''
        all_features = []

        # Create heatmap for the first 10 hidden units (originally 64)
        for i in range(X_test.shape[0]): 
            output = model(X_test[i].unsqueeze(0))  


        # Collect the activations from the model
        all_features.append(model.activations[0].cpu().numpy())  # Get activations for the current input and convert to CPU numpy array

        # Convert all features to a single numpy array
        all_features = np.array(all_features)

        # Visualize LSTM activations for all hidden units
        plt.figure(figsize=(12, 6))
        plt.imshow(all_features.T, aspect='auto', cmap='hot')  # Transpose for time steps along the x-axis and hidden units along the y-axis
        plt.title('LSTM Activations for the First Observation')
        plt.xlabel('Time Step')
        plt.ylabel('Hidden Unit Index')
        plt.colorbar(label='Activation Value')
        plt.savefig("lstm_activations_heatmap.png")
        plt.close()


    ###### actual vs predicted plots ######
    index = 6
    # Assume X_test is a tensor and you're accessing it by index
    tensor_sample = X_test[index]

    # Move the tensor sample to CPU, converting it to a NumPy array
    sample_np = tensor_sample.cpu().numpy()

    # Inverse transform using the scaler
    original_values = scaler_x.inverse_transform(sample_np.reshape(1, -1))  # Reshape to 2D array if necessary

    # Extract the azimuth and range values based on your dataset structure
    random_azimuth = original_values[0][0]  # First feature
    random_range = original_values[0][1]    # Second feature

    actual_values = y_test[index].cpu().numpy()
    actual_values = scaler_y.inverse_transform(actual_values)

    actual_azimuth, actual_bank = actual_values[0], actual_values[1]
    predicted_alpha, predicted_bank = predicting(random_azimuth, random_range, scaler_x, scaler_y)
    x_axis = np.arange(120)

    # Create a figure and an array of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column of subplots

    # First subplot for the first output
    axs[0].plot(x_axis, actual_azimuth, label='Actual', linestyle='-')
    axs[0].plot(x_axis, predicted_alpha, label='Predicted', linestyle='-')

    axs[0].set_title('Alpha')
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot for the second output
    axs[1].plot(x_axis, actual_bank, label='Actual', linestyle='-')
    axs[1].plot(x_axis, predicted_bank, label='Predicted', linestyle='-')
    axs[1].set_title('Bank')
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout to prevent overlap
    fig.suptitle(f'Comparison of Actual and Predicted Values\n(azimuth: {random_azimuth}, range: {random_range}, index: {index})')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_2.png')
    plt.close()  

    exit()






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
