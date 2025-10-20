def train_model(X_train, X_test, y_train, y_test, model, model_type, num_epochs, batch_size, patience, l1_lambda, learning_rate, window_size, device):
    model.to(device)
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
        # model.train()
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            if model_type == 'flex' or model_type == 'lstm_onetomany':
                loss = model.train(batch_X, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
            elif model_type == 'lstm_manytoone':
                loss = model.train(batch_X, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif model_type == 'transformer':
                loss = model.train(batch_X, batch_y, model)
                
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
                    loss = model.train(batch_X, batch_y)

                elif model_type == 'lstm_manytoone':
                    loss = model.train(batch_X, batch_y)

                elif model_type == 'transformer':
                    loss = model.train(batch_X, batch_y)

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

        if (epoch + 1) % 100 == 0:
            elapsed_time = time.time() - starttime
            print(f"Epoch {epoch + 1}/{num_epochs}: Train loss: {train_loss:.5E}, Test loss: {test_loss:.5E}, Time elapsed: {elapsed_time:.2f} sec")

    return model, train_losses, test_losses

if __name__ == "__main__":
    start_time = time.time()

    ###### Set Argument Parser ######
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-e", "--epochs", required=True, type=int, help="number of epochs")
    
    # args = parser.parse_args()
    # num_epochs = args.epochs

    ###### Set Parameters ######
    model_type = 'transformer'
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


    criterion = nn.MSELoss()
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
        
        model = LSTMModel_ManytoOne(input_dim, output_dim, hidden_size, num_layers, dropout, criterion).to(device)

    if model_type == 'transformer':
        model = TransformerCondDecoder(criterion=criterion).to(device)

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
    def custom_scorer(y_true, y_pred):
        y_pred_reshaped = y_pred.reshape(y_true.shape)
        return -np.mean((y_pred_reshaped - y_true) ** 2)

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
                 d_model=256, nhead=8, num_layers=6, dropout=0.1, 
                 criterion=None):
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
        self.criterion = criterion

    def forward(self, cond, y_prev):
        B, L, _ = y_prev.shape
        device = y_prev.device

        mem = self.cond_proj(cond).unsqueeze(1)            # [B, 1, d_model]
        tgt = self.y_in_proj(y_prev)                       # [B, L, d_model]
        tgt = self.pos(tgt)

        tgt_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)  # [L, L]
        h = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)  # [B, L, d_model]
        return self.out_head(h)                            # [B, L, 2]

    def train(self, batch_X, batch_y, model):
        B, C, L = batch_y.shape
        target  = batch_y.permute(0,2,1)
        start = model.start_token.expand(B, 1, C).to(batch_y.device)
        y_prev = torch.cat([start, target[:, :-1, :]], dim=1)
        preds = model(batch_X, y_prev)
        loss = self.criterion(preds, target)
        return loss

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

        
