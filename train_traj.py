class CurveSensitiveLoss(nn.Module):
    def __init__(self, mse_weight=1.0, derivative_weight=0.5, curvature_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.derivative_weight = derivative_weight
        self.curvature_weight = curvature_weight
        self.mse = nn.MSELoss()
        
        # Store components for logging (no gradient tracking)
        self.last_mse = 0
        self.last_derivative = 0
        self.last_curvature = 0
    
    def forward(self, pred, target):
        # Basic MSE loss
        mse_loss = self.mse(pred, target)
        
        # First derivative loss
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        target_diff = target[:, 1:, :] - target[:, :-1, :]
        derivative_loss = self.mse(pred_diff, target_diff)
        
        # Second derivative loss
        pred_diff2 = pred_diff[:, 1:, :] - pred_diff[:, :-1, :]
        target_diff2 = target_diff[:, 1:, :] - target_diff[:, :-1, :]
        curvature_loss = self.mse(pred_diff2, target_diff2)
        
        # Store for logging (detached from graph)
        self.last_mse = mse_loss.item()
        self.last_derivative = derivative_loss.item()
        self.last_curvature = curvature_loss.item()
        
        # Return single scalar loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.derivative_weight * derivative_loss +
                     self.curvature_weight * curvature_loss)
        
        return total_loss
    
    def get_components(self):
        """Get last computed loss components (for logging)"""
        return {
            'mse': self.last_mse,
            'derivative': self.last_derivative,
            'curvature': self.last_curvature
        }


# ============================================================================
# POSITIONAL ENCODING (for Transformer)
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)

# ============================================================================
# TRANSFORMER MODEL
# ============================================================================
class TransformerCondDecoder(nn.Module):
    def __init__(self, seq_len=106, cond_dim=2, out_dim=2,
                 d_model=512, nhead=8, num_layers=6, dropout=0.1, criterion=None):
        super(TransformerCondDecoder, self).__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.y_in_proj = nn.Linear(out_dim, d_model)
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
        self.device = None

    def _causal_mask(self, T: int, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, cond, y_prev):
        B, L, _ = y_prev.shape
        device = y_prev.device
        self.device = device
        mem = self.cond_proj(cond).unsqueeze(1)
        tgt = self.y_in_proj(y_prev)
        tgt = self.pos(tgt)
        tgt_mask = self._causal_mask(L, device)
        h = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)
        return self.out_head(h)

    def compute_loss(self, batch_X, batch_y):
        """
        batch_X: [B, 2]
        batch_y: [B, 2, 106] -> convert to [B, 106, 2]
        """
        if batch_y.dim() == 3 and batch_y.shape[1] == self.out_dim:
            target = batch_y.permute(0, 2, 1)  # [B, 2, L] -> [B, L, 2]
        else:
            target = batch_y
        
        B = batch_X.size(0)
        start = self.start_token.expand(B, 1, self.out_dim).to(batch_X.device)
        y_prev = torch.cat([start, target[:, :-1, :]], dim=1)
        
        preds = self.forward(batch_X, y_prev)
        loss = self.criterion(preds, target)
        return loss

    @torch.no_grad()
    def predict_batch(self, batch_X):
        """Generate sequences for batch"""
        self.eval()
        device = batch_X.device
        self.device = device
        B = batch_X.size(0)
        L = self.seq_len
        mem = self.cond_proj(batch_X).unsqueeze(1)
        y_tokens = self.start_token.expand(B, 1, self.out_dim).to(device)
        for _ in range(L):
            tgt = self.y_in_proj(y_tokens)
            tgt = self.pos(tgt)
            tgt_mask = self._causal_mask(tgt.size(1), device)
            h = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)
            step = self.out_head(h[:, -1:, :])
            y_tokens = torch.cat([y_tokens, step], dim=1)
        return y_tokens[:, 1:, :]  # [B, L, 2]

    def predict(self, azimuth, range_, scaler_x, scaler_y):
        L=106
        self.to(self.device)
        self.eval() 
        input_data = np.array([[azimuth, range_]], dtype=np.float32)
        scaled_input = scaler_x.transform(input_data)
        batch_X_test = torch.tensor(scaled_input, dtype=torch.float32).to(self.device)

        seq = self.predict_batch(batch_X_test)
        alpha_scaled = seq[0, :, 0].reshape(1,L).cpu().numpy()
        bank_scaled = seq[0, :, 1].reshape(1,L).cpu().numpy()

        alpha = scaler_y.inverse_transform(alpha_scaled)[0].tolist()
        bank = scaler_y.inverse_transform(bank_scaled)[0].tolist()

        return alpha, bank

    for epoch in range(num_epochs):
        # ============ TRAINING ============
        model.train()
        epoch_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(batch_X, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ============ VALIDATION ============
        model.eval()
        epoch_test_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                loss = model.compute_loss(batch_X, batch_y)
                epoch_test_loss += loss.item()
        
        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Learning rate scheduling
