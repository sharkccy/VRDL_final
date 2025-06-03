import torch
import torch.nn as nn
import timm
import torchvision

class SwinRegression(nn.Module):
    def __init__(self, num_classes=5):
        super(SwinRegression, self).__init__()
        # 載入 Swin-Tiny 預訓練模型，啟用 features_only
        self.swin = timm.create_model(
            'swinv2_tiny_window8_256.ms_in1k',
            pretrained=True,
            features_only=True
        )
        
        # 預設凍結所有參數
        self.freeze_parameters()

        # 回歸層，直接將特徵圖展平
        self.regressor = nn.Sequential(
            nn.Flatten(),  # 將 (batch_size, 768, 7, 7) 展平為 (batch_size, 768*7*7)
            nn.Linear(768 * 8 * 8, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),  # 輸出 5 個值
        )

    def freeze_parameters(self):
        """凍結所有 Swin Transformer 的參數"""
        # for param in self.swin.parameters():
        #     param.requires_grad = False
        
        for name, param in self.swin.named_parameters():
            # 如果不是最後一層的參數，則凍結
            if 'layers.3' not in name:  # layers.3 是最後一層
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_parameters(self):
        """解凍所有 Swin Transformer 的參數"""
        for param in self.swin.parameters():
            param.requires_grad = True

    def forward(self, x):
        # 獲取所有階段的特徵圖
        features = self.swin(x)
        
        # 取最後一階段的特徵圖 (batch_size, 7, 7, 768)
        x = features[-1]
        
        # 轉置維度從 NHWC 到 NCHW
        x = x.permute(0, 3, 1, 2)  # (batch_size, 768, 7, 7)
        
        # 直接通過回歸層（包含展平操作）
        x = self.regressor(x)
        return x
    
class VggRegression(nn.Module):
    def __init__(self, num_classes=5, input_size=(300, 300)):
        super(VggRegression, self).__init__()
        
        # 載入預訓練的 VGG16
        self.vgg = torchvision.models.vgg16(pretrained=True)
        
        # 定義自訂的回歸層，替換 classifier
        self.vgg.classifier = nn.Sequential(
            nn.Flatten(),                    # 展平特徵圖 (512, 9, 9) -> (512 * 9 * 9)
            nn.Linear(512 * 9 * 9, 1024),    # 對應 TensorFlow 的 Dense(1024)
            nn.LeakyReLU(negative_slope=0.1),# LeakyReLU with alpha=0.1
            nn.Dropout(0.5),                 # 可選的正則化
            nn.Linear(1024, num_classes)     # 5 個輸出，線性激活
        )

    def forward(self, x):
        # 直接通過 VGG16（包括自訂的 classifier）
        x = self.vgg(x)
        return x

# 測試程式碼
if __name__ == "__main__":
    # 模擬輸入張量 (batch_size, channels, height, width)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 300, 300)
    
    # 初始化模型
    model = VggRegression(num_classes=5)
    model.eval()  # 設置為評估模式
    
    # 前向傳播
    with torch.no_grad():
        output = model(input_tensor)
        print(f"輸出形狀: {output.shape}")  # 應為 (batch_size, 5)