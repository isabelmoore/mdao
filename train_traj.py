import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

# ============================================================================
# BASE MODEL INTERFACE
# ============================================================================

class BaseModelInterface(nn.Module):
    """Base class that all models should inherit from for unified training"""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def compute_loss(self, batch_X, batch_y):
        """
        Compute loss for the model - must be implemented by each model
        Args:
            batch_X: [B, input_dim] - input conditions
            batch_y: [B, C, L] or [B, L, C] - target sequences
        Returns:
            loss: scalar tensor
        """
        raise NotImplementedError("Each model must implement compute_loss()")
    
    def predict_batch(self, batch_X):
        """
        Generate predictions for a batch
        Args:
            batch_X: [B, input_dim]
        Returns:
            predictions: [B, L, C] or appropriate output format
        """
        raise NotImplementedError("Each model must implement predict_batch()")

# ============================================================================
# POSITIONAL ENCODING (for Transformer)
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
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

class TransformerCondDecoder(BaseModelInterface):
    def __init__(self, seq_len=106, cond_dim=2, out_dim=2,
                 d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
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

    def _causal_mask(self, T: int, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, cond, y_prev):
        B, L, _ = y_prev.shape
        device = y_prev.device

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

# ============================================================================
# FLEXIBLE NEURAL NETWORK
# ============================================================================

class FlexibleNeuralNet(BaseModelInterface):
    def __init__(self, layer_sizes, activation_type="ReLU", dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Don't add activation after last layer
                if activation_type == "ReLU":
                    self.layers.append(nn.ReLU())
                elif activation_type == "tanh":
                    self.layers.append(nn.Tanh())
                self.layers.append(nn.Dropout(dropout))
        
        self.output_shape = (2, 106)  # (channels, sequence_length)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def compute_loss(self, batch_X, batch_y):
        """
        batch_X: [B, 2]
        batch_y: [B, 2, 106]
        """
        preds = self.forward(batch_X)  # [B, 212]
        target = batch_y.reshape(batch_y.size(0), -1)  # [B, 212]
        loss = self.criterion(preds, target)
        return loss
    
    @torch.no_grad()
    def predict_batch(self, batch_X):
        """Generate predictions"""
        self.eval()
        preds = self.forward(batch_X)  # [B, 212]
        # Reshape to [B, L, C]
        B = preds.size(0)
        preds = preds.reshape(B, self.output_shape[1], self.output_shape[0])
        return preds

# ============================================================================
# LSTM ONE-TO-MANY
# ============================================================================

class LSTMModel_OnetoMany(BaseModelInterface):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, dropout=0.5, seq_len=106):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x):
        """
        x: [B, input_dim]
        Returns: [B, seq_len, output_dim]
        """
        B = x.size(0)
        x = x.unsqueeze(1)  # [B, 1, input_dim]
        
        h0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size).to(x.device)
        
        outputs = []
        h, c = h0, c0
        
        for _ in range(self.seq_len):
            out, (h, c) = self.lstm(x, (h, c))
            pred = self.fc(out)  # [B, 1, output_dim]
            outputs.append(pred)
        
        return torch.cat(outputs, dim=1)  # [B, seq_len, output_dim]
    
    def compute_loss(self, batch_X, batch_y):
        """
        batch_X: [B, 2]
        batch_y: [B, 2, 106] -> convert to [B, 106, 2]
        """
        if batch_y.dim() == 3 and batch_y.shape[1] == 2:
            target = batch_y.permute(0, 2, 1)
        else:
            target = batch_y
        
        preds = self.forward(batch_X)
        loss = self.criterion(preds, target)
        return loss
    
    @torch.no_grad()
    def predict_batch(self, batch_X):
        self.eval()
        return self.forward(batch_X)

# ============================================================================
# UNIFIED TRAINING FUNCTION
# ============================================================================

def train_model(model, X_train, X_test, y_train, y_test, 
                num_epochs=1000, batch_size=64, learning_rate=0.001,
                patience=50, device='cuda', print_every=100):
    """
    Unified training function that works with any model inheriting from BaseModelInterface
    
    Args:
        model: Instance of a model class inheriting from BaseModelInterface
        X_train, X_test: Training and test inputs [N, input_dim]
        y_train, y_test: Training and test targets [N, C, L]
        num_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
        device: Device to train on
        print_every: Print progress every N epochs
    
    Returns:
        model: Trained model
        train_losses: List of training losses
        test_losses: List of test losses
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=patience//2)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
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
        scheduler.step(avg_test_loss)
        
        # ============ EARLY STOPPING ============
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
            }, 'best_model.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        # ============ LOGGING ============
        if (epoch + 1) % print_every == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.5e} | "
                  f"Test Loss: {avg_test_loss:.5e} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.2f}s")
    
    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nTraining completed. Best test loss: {best_test_loss:.5e}")
    
    return model, train_losses, test_losses

# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type, config):
    """
    Factory function to create models based on type
    
    Args:
        model_type: str - 'transformer', 'flex', 'lstm_onetomany', etc.
        config: dict - configuration parameters for the model
    
    Returns:
        model: Instance of the requested model
    """
    if model_type == 'transformer':
        return TransformerCondDecoder(
            seq_len=config.get('seq_len', 106),
            cond_dim=config.get('cond_dim', 2),
            out_dim=config.get('out_dim', 2),
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            dropout=config.get('dropout', 0.1)
        )
    
    elif model_type == 'flex':
        input_size = config.get('input_size', 2)
        output_size = config.get('output_size', 212)  # 2 * 106
        hidden_size = config.get('hidden_size', 64)
        num_layers = config.get('num_layers', 3)
        
        layer_sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        
        return FlexibleNeuralNet(
            layer_sizes=layer_sizes,
            activation_type=config.get('activation_type', 'ReLU'),
            dropout=config.get('dropout', 0.5)
        )
    
    elif model_type == 'lstm_onetomany':
        return LSTMModel_OnetoMany(
            input_dim=config.get('input_dim', 2),
            output_dim=config.get('output_dim', 2),
            hidden_size=config.get('hidden_size', 64),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.5),
            seq_len=config.get('seq_len', 106)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example data (replace with your actual data loading)
    # X_train, X_test, y_train, y_test = load_your_data()
    
    # Model configuration
    model_configs = {
        'transformer': {
            'seq_len': 106,
            'cond_dim': 2,
            'out_dim': 2,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dropout': 0.1
        },
        'flex': {
            'input_size': 2,
            'output_size': 212,
            'hidden_size': 64,
            'num_layers': 3,
            'activation_type': 'ReLU',
            'dropout': 0.5
        },
        'lstm_onetomany': {
            'input_dim': 2,
            'output_dim': 2,
            'hidden_size': 64,
            'num_layers': 3,
            'dropout': 0.5,
            'seq_len': 106
        }
    }
    
    # Choose model type
    model_type = 'transformer'  # or 'flex', 'lstm_onetomany'
    
    # Create model
    model = create_model(model_type, model_configs[model_type])
    print(f"\n{model_type.upper()} Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    # model, train_losses, test_losses = train_model(
    #     model=model,
    #     X_train=X_train,
    #     X_test=X_test,
    #     y_train=y_train,
    #     y_test=y_test,
    #     num_epochs=10000,
    #     batch_size=64,
    #     learning_rate=0.001,
    #     patience=500,
    #     device=device,
    #     print_every=100
    # )
    
    # Save final model
    # torch.save(model.state_dict(), f'{model_type}_final_model.pt')
