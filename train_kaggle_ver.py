import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchmetrics import MeanSquaredError

from util.dataset import SeaLionPatchDataset
from util.schedulers import LinearWarmupCosineAnnealingLR
from net.model_kaggle_ver import SwinRegression


torch.set_float32_matmul_precision('medium')  

# 定義 Lightning 模組
class SeaLionCountingModel(pl.LightningModule):
    def __init__(self, num_classes=5, learning_rate=0.0001, batch_size=8):
        super().__init__()
        self.save_hyperparameters()
        self.model = SwinRegression(num_classes)
        self.criterion = nn.MSELoss()
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.train_rmse(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_rmse', self.train_rmse, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.val_rmse(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_rmse', self.val_rmse, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=10,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=0.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }  

# 定義數據模組
class SeaLionDataModule(pl.LightningDataModule):
    def __init__(self, train_files, train_dir, dotted_dir, batch_size=8, patch_size = 224, num_workers=0, scale_factor=0.4):
        super().__init__()
        self.train_files = train_files
        self.train_dir = train_dir
        self.dotted_dir = dotted_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers  
        self.scale_factor = scale_factor
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])
    def setup(self, stage=None):
        if stage in (None, "fit"):
            # 創建完整的數據集
            full_dataset = SeaLionPatchDataset(
                self.train_files, self.train_dir, self.dotted_dir, self.patch_size, self.scale_factor,
                self.transform  
            )
            # 按 80% 訓練，20% 驗證分割
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

if __name__ == "__main__":
    # 解析命令列參數
    parser = argparse.ArgumentParser(description="Train Sea Lion Counting Model")
    parser.add_argument("--train_dir", type=str, default="Train", help="Path to train directory")
    parser.add_argument("--dotted_dir", type=str, default="TrainDotted", help="Path to dotted directory")
    parser.add_argument("--save_model", type=str, default="output", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the patches to extract from images")
    parser.add_argument("--scale_factor", type=float, default=0.4, help="Scale factor for image patches")

    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    # 準備數據
    print("Preparing dataset...")
    train_files = [f for f in os.listdir(args.train_dir) if f.endswith('.png') or f.endswith('.jpg')]
    data_module = SeaLionDataModule(train_files, args.train_dir, args.dotted_dir,
                                    batch_size=args.batch_size, patch_size=args.patch_size, num_workers=args.num_workers, scale_factor=args.scale_factor)
    
    print("Initializing model...")
    # 初始化模型
    model = SeaLionCountingModel(num_classes=5, learning_rate=args.lr, batch_size=args.batch_size)

    print("Setting up training...")
    # 設置 Logger
    logger = TensorBoardLogger("lightning_logs", name="sea_lion_counting")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_model,
        filename="swin_pytorch-{epoch:02d}-{val_rmse:.4f}",
        monitor="val_rmse",
        mode="min",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1
    )

    # 訓練
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    # 儲存模型
    # torch.save(model.state_dict(), os.path.join(args.save_model, "sea_lion_counting_model.pth"))
