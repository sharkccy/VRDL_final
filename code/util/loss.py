import torch
import torch.nn as nn

class FocalRMSELoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, eps=1e-6, squared=True):
        super(FocalRMSELoss, self).__init__()
        self.alpha = alpha if alpha is not None else torch.ones(5)  # 5 個類別
        self.gamma = gamma
        self.eps = eps
        self.squared = squared

    def forward(self, pred, target):
        alpha = self.alpha.to(pred.device) if isinstance(self.alpha, torch.Tensor) else self.alpha
        error = torch.abs(target - pred)  # [batch_size, 5]
        squared_error = error ** 2  # [batch_size, 5]
        focal_weight = alpha.unsqueeze(0) * (1 + error) ** self.gamma  # [batch_size, 5]
        weighted_loss = focal_weight * squared_error
        mean_loss = weighted_loss.mean()
        if self.squared:
            rmse = torch.sqrt(mean_loss + self.eps)
        else:
            rmse = mean_loss + self.eps
        return rmse