import torch
import torch.nn as nn
import timm

class SwinRegression(nn.Module):
    def __init__(self, num_classes=5):
        super(SwinRegression, self).__init__()
        # 載入 Swin-Tiny 預訓練模型
        self.swin = timm.create_model('swin_tiny_patch4_window7_224.ms_in22k', pretrained=True)
        for param in self.swin.parameters():
            param.requires_grad = False
            
        self.regressor = nn.Sequential(
            nn.Linear(768, 512),  # Swin-Tiny 輸出 768 維
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # timm 的 Swin 模型直接輸出全局特徵 (batch_size, 768)
        x = self.swin(x)  # 移除 pixel_values 和 last_hidden_state 相關操作
        x = self.regressor(x)
        return x