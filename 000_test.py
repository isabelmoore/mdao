
def train_model_transformer(X_train, X_test, y_train, y_test, model, num_epochs, batch_size, patience, l1_lambda, learning_rate, window_size, device):
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

def train_model_manytoone(X_train, X_test, y_train, y_test, model, num_epochs, batch_size, patience, l1_lambda, learning_rate, window_size, device):
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
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            window_y = torch.zeros(batch_size, batch_y.size(2), window_size).to(device) 

            for start in range(0, seq_length, window_size):  
                end = start + window_size
                
                if end > seq_length:
                    end = seq_length
                current_window_len = end - start

                if start != 0:
                    window_y = torch.cat((window_y[:, :, 1:], final_prediction.unsqueeze(2)), dim=2)

                window_X = batch_X.unsqueeze(2).expand(-1, -1, window_size)

                window_X = window_X[:, :, start:end]  
                window_y_target = batch_y[:, :, start:end] 
 
                predictions = model(window_X)

                loss = criterion(predictions, window_y_target)

                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l1_loss = l1_lambda * l1_norm
                loss += l1_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                final_prediction = predictions[:, -1, :].detach()  
                train_loss += loss.item()  

        train_loss /= len(data_loader)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        model.eval()
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if (epoch + 1) % 1 == 0:
            elapsed_time = time.time() - starttime
            print(f"Epoch {epoch + 1}/{num_epochs}: Train loss: {train_loss:.5E}, Time elapsed: {elapsed_time:.2f} sec")

    return model, train_losses, test_losses

def train_model_onetomany(X_train, X_test, y_train, y_test, model, num_epochs, batch_size, patience, l1_lambda, learning_rate, window_size, device):
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
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = batch_X.unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = model.criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # Accumulate training loss for reporting
                
            train_loss += loss.item()
        train_loss /= len(data_loader)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        model.eval()
        dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            test_loss = 0

            for batch_X, batch_y in test_loader:
                batch_X_test, batch_y_test = batch_X.to(device), batch_y.to(device)
                batch_X_test = batch_X_test.unsqueeze(1)

                optimizer.zero_grad()
                predictions = model(batch_X_test)
                loss = model.criterion(predictions, batch_y_test)
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
        return x.view(x.size(0), 2, 1)  # Assure shape compatibility


class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, nhead, max_len):
        super(TransformerSeq2Seq, self).__init__()
        # Transformer layers
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_size, nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        
        # Fully connected layers for input and output
        self.input_fc = nn.Linear(input_dim, hidden_size)  # Map inputs to hidden size
        self.output_fc = nn.Linear(hidden_size, output_dim)  # Map hidden reps to outputs
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: [batch_size, src_seq_length, input_dim] – Input sequence
            tgt: [batch_size, tgt_seq_length, output_dim] – Target sequence for prediction
            src_mask: Optional, mask for source padding
            tgt_mask: Optional, mask for preventing attention to future tokens
        Returns:
            output: [batch_size, tgt_seq_length, output_dim]
        """
        # Transform inputs to the model's hidden size
        src = self.input_fc(src).permute(1, 0, 2)  # [src_seq_length, batch_size, hidden_size]
        tgt = self.input_fc(tgt).permute(1, 0, 2)  # [tgt_seq_length, batch_size, hidden_size]
        
        # Encoding and decoding steps
        memory = self.encoder(src, src_key_padding_mask=src_mask)  # Encode input
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)  # Decode predictions
        
        # Project back to output dimensionality
        return self.output_fc(output.permute(1, 0, 2))  # [batch_size, tgt_seq_length, output_dim]


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
    
