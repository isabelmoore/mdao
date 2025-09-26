            # Autoregressive generate: [1, L, 2]
            preds = self.model.generate(cond, L=L).cpu().numpy()
        else:
            # Direct forward: get [1,2,L] (some models output [1,2,L] already)
            out = self.model(cond)
            if out.dim() == 2:      # [1,2] -> expand to [1,2,L]
                out = out.unsqueeze(-1).expand(-1, -1, L)
            preds = out.permute(0, 2, 1).cpu().numpy()   # [1,L,2]

        # Inverse transform Y using scaler_y
        B, T, C = preds.shape  # [1,L,2]
        flat = preds.reshape(B, T*C)
        inv_flat = scaler_y.inverse_transform(flat)
        inv = inv_flat.reshape(B, T, C)  # [1,L,2]

        alpha = inv[0, :, 0].tolist()   # length L
        bank  = inv[0, :, 1].tolist()   # length L
        return alpha, bank
