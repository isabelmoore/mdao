
class FlexibleNeuralNet(nn.Module):
    def __init__(self, num_layers, activation_type="tanh", dropout=0.5):
        super(FlexibleNeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        input_size = 2  
        output_size = 212 
        output_shape = (2, 106)
        layer_sizes = [input_size] + [hidden_size] * num_layers + [output_size]  
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
        self.model = model
        self.device = device

    def forward(self, x):
        # Flatten the input for the fully connected layers
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x.view(x.size(0), 2, -1)  # Assure shape compatibility

    def train(self, batch_X, batch_y):
        batch_X = batch_X.unsqueeze(1)
        predictions = self.model(batch_X)
        loss = criterion(predictions, batch_y)
        return loss
    
    def predict(self, azimuth, range_, scaler_x, scaler_y):
        self.model.to(device)
        self.model.eval() 
        input_data = np.array([[azimuth, range_]], dtype=np.float32)
        scaled_input = scaler_x.transform(input_data)
        batch_X_test = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(1).to(self.device)
         
        predictions = self.model(batch_X_test)
        predictions_cpu = predictions.cpu().numpy().reshape(-1, 106)
        
        inverse_predictions = scaler_y.inverse_transform(predictions_cpu) 
        return inverse_predictions[0], inverse_predictions[1]

class LSTMModel_OnetoMany(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout=0.5):
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
        self.model = model
        self.device = device

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        lstm_out = lstm_out.permute(0, 2, 1)  
        output = self.final_output(lstm_out) 
        output = output.view(-1, self.output_dim, self.output_length)
        return output
    
    def train(self, batch_X, batch_y):
        batch_X = batch_X.unsqueeze(1)
        predictions = self.model(batch_X)
        loss = criterion(predictions, batch_y)
        return loss
    
    def predict(self, azimuth, range_, scaler_x, scaler_y):
        self.model.to(device)
        self.model.eval() 
        input_data = np.array([[azimuth, range_]], dtype=np.float32)
        scaled_input = scaler_x.transform(input_data)
        batch_X_test = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(1).to(self.device)
         
        predictions = self.model(batch_X_test)
        predictions_cpu = predictions.cpu().numpy().reshape(-1, 106)
        
        inverse_predictions = scaler_y.inverse_transform(predictions_cpu) 
        return inverse_predictions[0], inverse_predictions[1]

class LSTMModel_ManytoOne(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout=0.5):
        super(LSTMModel_ManytoOne, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = 4
        self.output_dim = 2

        self.lstm = nn.LSTM( self.input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.final_output = nn.Linear(hidden_size,  self.output_dim)
        self.model = model
        self.device = device

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        lstm_out = lstm_out[:, -1, :] 
        output = self.final_output(lstm_out) 
        return output 

    def train(self, batch_X, batch_y):
        predictions = []
        targets = []

        for start in range(0, seq_length + 1):
            prediction, target = nnm.train_manytoone(start, window_size, seq_length, batch_X, batch_y)
            
            predictions.append(prediction)  
            targets.append(target)  
            
        predictions = torch.cat(predictions, dim=1)  
        targets = torch.cat(targets, dim=1)

        loss = criterion(predictions, targets)
        return loss

    def predict(self, azimuth, range_, scaler_x, scaler_y):
        self.model.to(device)
        self.model.eval() 

        input_data = np.array([[azimuth, range_]], dtype=np.float32)
        scaled_input = scaler_x.transform(input_data)
        batch_X_test = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(1).to(self.device)
        
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)

class TransformerCondDecoder(nn.Module):
    def __init__(self, seq_len=120, cond_dim=2, out_dim=2,
                 d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(TransformerCondDecoder, self).__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim

        self.cond_proj = nn.Linear(cond_dim, d_model)      # condition -> memory token
        self.y_in_proj = nn.Linear(out_dim, d_model)       # previous outputs -> tokens
        self.pos = PositionalEncoding(d_model, max_len=seq_len+1)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model, dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_head = nn.Linear(d_model, out_dim)
        self.start_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def forward(self, cond, y_prev):
        B, L, _ = y_prev.shape
        device = y_prev.device

        mem = self.cond_proj(cond).unsqueeze(1)            # [B, 1, d_model]
        tgt = self.y_in_proj(y_prev)                       # [B, L, d_model]
        tgt = self.pos(tgt)

        tgt_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)  # [L, L]
        h = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)  # [B, L, d_model]
        return self.out_head(h)                            # [B, L, 2]

    def train(self, batch_X, batch_y):
        B, C, L = batch_y.shape
        target  = batch_y.permute(0,2,1)
        start = model.start_token.expand(B, 1, C).to(batch_y.device)
        y_prev = torch.cat([start, target[:, :-1, :]], dim=1)
        preds = model(batch_X, y_prev)
        loss = criterion(preds, target)
        return loss()

    def generate(self, cond, L=None):
        self.eval()
        device = cond.device
        L = L or self.seq_len
        B = cond.size(0)

        mem = self.cond_proj(cond).unsqueeze(1)            # [B,1,d_model]
        y_tokens = self.start_token.expand(B, 1, self.out_dim).to(device)  # [B,1,2]

        for _ in range(L):
            tgt = self.y_in_proj(y_tokens)                 # [B,t, d_model]
            tgt = self.pos(tgt)
            tgt_mask = self._causal_mask(tgt.size(1), device)
            h = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)
            step = self.out_head(h[:, -1:, :])             # [B,1,2]
            y_tokens = torch.cat([y_tokens, step], dim=1)

        return y_tokens[:, 1:, :]                          # [B,L,2]

    def predict(self, azimuth, range_, scaler_x, scaler_y):
        L=106
        batch_X_test = torch.tensor(scaled_input, dtype=torch.float32).to(self.device)

        seq = self.generate(batch_X_test, L)
        alpha_scaled = seq[0, :, 0].reshape(1,L).cpu().numpy()
        bank_scaled = seq[0, :, 1].reshape(1,L).cpu().numpy()

        alpha = scaler_y.inverse_transform(alpha_scaled)[0].tolist()
        bank = scaler_y.inverse_transform(bank_scaled)[0].tolist()

        return alpha, bank
