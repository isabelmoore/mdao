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
