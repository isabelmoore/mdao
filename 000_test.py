class NN_Models():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)   

    def train_onetomany(self, batch_X, batch_y):
        batch_X = batch_X.unsqueeze(1)
        predictions = self.model(batch_X)
        return predictions
    
    def train_manytoone(self, start, window_size, seq_length, batch_X, batch_y):
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
                loss = model.criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                    
            if model_type == 'lstm_manytoone':
                window_y = torch.zeros(batch_size, batch_y.size(2), window_size).to(device) 

                for start in range(0, seq_length, window_size):  
                    predictions, window_y_target = nnm.train_manytoone(start, window_size, seq_length, batch_X, batch_y)
                    loss = criterion(predictions, window_y_target)

                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    l1_loss = l1_lambda * l1_norm
                    loss += l1_loss

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



    if model_type == 'flex':
        input_size = 2  
        output_size = 240 
        output_shape = (2, 120)

        activation_type = "ReLU"

        layer_sizes = [input_size] + [hidden_size] * num_layers + [output_size]  
        print(f"Layer Sizes: {layer_sizes}")
        
        model = FlexibleNeuralNet(layer_sizes=layer_sizes, activation_type=activation_type, dropout=dropout).to(device)

    if model_type == 'lstm_onetomany':
        input_dim = 2
        output_dim = 2
        
        model = LSTMModel_OnetoMany(input_dim, output_dim, hidden_size, num_layers, dropout).to(device)

    if model_type == 'lstm_manytoone':
        input_dim = 2
        output_dim = 2
        
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
