import torch
import torch.nn as nn
import timm

class SwinRegression(nn.Module):
    def __init__(self, num_classes=5):
        super(SwinRegression, self).__init__()
        # 載入 Swin-Tiny 預訓練模型，啟用 features_only
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224.ms_in22k',
            pretrained=True,
            features_only=True
        )
        for param in self.swin.parameters():
            param.requires_grad = False
            
        # 全局平均池化層，將 (batch_size, C, H, W) 轉為 (batch_size, C, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 回歸層，輸入維度為 768 (Swin-Tiny 最後一階段的通道數)
        self.regressor = nn.Sequential(
            nn.Linear(768, 512),  # 從 768 維到 512 維
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 輸出 5 個值
        )

    def forward(self, x):
        # 獲取所有階段的特徵圖
        features = self.swin(x)
        # print("Feature map shapes:")
        # for i, f in enumerate(features):
            # print(f"Stage {i} shape: {f.shape}")
        
        # 取最後一階段的特徵圖 (batch_size, 7, 7, 768)
        x = features[-1]  # 形狀為 (batch_size, 7, 7, 768)
        
        # 轉置維度從 NHWC 到 NCHW
        x = x.permute(0, 3, 1, 2)  # (batch_size, 768, 7, 7)
        # print(f"Shape after permute: {x.shape}")
        
        # 全局平均池化：(batch_size, 768, 7, 7) -> (batch_size, 768, 1, 1)
        x = self.pool(x)
        # print(f"Shape after pool: {x.shape}")
        
        # 展平：(batch_size, 768, 1, 1) -> (batch_size, 768)
        x = x.view(x.size(0), -1)
        # print(f"Shape after pooling and flattening: {x.shape}")
        
        # 通過回歸層
        x = self.regressor(x)
        return x