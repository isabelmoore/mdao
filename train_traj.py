import math
import torch
import torch.nn as nn

# --- positional encoding ---
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

# --- decoder-only transformer conditioned on (azimuth, range) ---
class TransformerCondDecoder(nn.Module):
    def __init__(self, seq_len=120, cond_dim=2, out_dim=2,
                 d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
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

        # learned start token (y[-1])
        self.start_token = nn.Parameter(torch.zeros(1, 1, out_dim))

    def _causal_mask(self, T: int, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, cond, y_prev):
        """
        cond:   [B, 2]
        y_prev: [B, L, 2]  (teacher-forced inputs: start + y[:, :-1, :])
        return: [B, L, 2]
        """
        B, L, _ = y_prev.shape
        device = y_prev.device

        mem = self.cond_proj(cond).unsqueeze(1)            # [B, 1, d_model]
        tgt = self.y_in_proj(y_prev)                       # [B, L, d_model]
        tgt = self.pos(tgt)

        tgt_mask = self._causal_mask(L, device)            # [L, L]
        h = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)  # [B, L, d_model]
        return self.out_head(h)                            # [B, L, 2]

    @torch.no_grad()
    def generate(self, cond, L=None):
        """
        cond: [B, 2] -> returns [B, L, 2]
        """
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

