# Problem: MSE penalizes all errors equally
# Result: Model learns to predict the "mean" trajectory, missing curves

# Solution: Use multiple loss components
class CurveSensitiveLoss(nn.Module):
    def __init__(self, mse_weight=1.0, derivative_weight=0.5, curvature_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.derivative_weight = derivative_weight
        self.curvature_weight = curvature_weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        pred, target: [B, L, C] where L=106, C=2
        """
        # Basic MSE loss
        mse_loss = self.mse(pred, target)
        
        # First derivative loss (velocity/rate of change)
        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        target_diff = target[:, 1:, :] - target[:, :-1, :]
        derivative_loss = self.mse(pred_diff, target_diff)
        
        # Second derivative loss (acceleration/curvature)
        pred_diff2 = pred_diff[:, 1:, :] - pred_diff[:, :-1, :]
        target_diff2 = target_diff[:, 1:, :] - target_diff[:, :-1, :]
        curvature_loss = self.mse(pred_diff2, target_diff2)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.derivative_weight * derivative_loss +
                     self.curvature_weight * curvature_loss)
        
        return total_loss, {
            'mse': mse_loss.item(),
            'derivative': derivative_loss.item(),
            'curvature': curvature_loss.item()
        }

# Usage in your model
class TransformerCondDecoder(BaseModelInterface):
    def __init__(self, ...):
        super().__init__()
        # ... existing init ...
        self.criterion = CurveSensitiveLoss(
            mse_weight=1.0,
            derivative_weight=0.5,  # Emphasize velocity matching
            curvature_weight=0.3    # Emphasize curve matching
        )
    
    def compute_loss(self, batch_X, batch_y):
        if batch_y
