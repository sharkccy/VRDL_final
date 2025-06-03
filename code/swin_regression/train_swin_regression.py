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
import pandas as pd

from util.dataset import SeaLionPatchDataset
from util.schedulers import LinearWarmupCosineAnnealingLR
from util.loss import FocalRMSELoss
from net.model_kaggle_ver import SwinRegression, VggRegression

torch.set_float32_matmul_precision('medium')  

# Define Lightning module
class SeaLionCountingModel(pl.LightningModule):
    def __init__(self, num_classes=5, learning_rate=0.0001, batch_size=8, alpha=None, gamma=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = SwinRegression(num_classes)
        self.criterion = FocalRMSELoss(alpha=alpha, gamma=gamma, squared=False)
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        # For accumulating training set counts
        self.train_ground_truths = []
        self.train_predictions = []
        # For accumulating validation set counts
        self.val_ground_truths = []
        self.val_predictions = []

    # def on_train_epoch_start(self):
    #     # 在第10個epoch時解凍所有參數
    #     if self.current_epoch == 10:
    #         print("第10個epoch開始，解凍所有參數")
    #         self.model.unfreeze_parameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.train_rmse(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_rmse', self.train_rmse, on_step=True, on_epoch=True, prog_bar=True)
        # Collect training set counts
        self.train_ground_truths.append(labels.detach().cpu())
        self.train_predictions.append(outputs.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.val_rmse(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_rmse', self.val_rmse, on_step=True, on_epoch=True, prog_bar=True)
        # Collect validation set counts
        self.val_ground_truths.append(labels.detach().cpu())
        self.val_predictions.append(outputs.detach().cpu())

    def on_train_epoch_end(self):
        # Aggregate training set counts
        ground_truths = torch.cat(self.train_ground_truths, dim=0)  # [N, 5]
        predictions = torch.cat(self.train_predictions, dim=0)  # [N, 5]
        gt_sum = ground_truths.sum(dim=0).round().int().numpy()  # [5]
        pred_sum = predictions.sum(dim=0).round().int().numpy()  # [5]
        print(f"\nTraining set --")
        print(f"    ground truth:  {gt_sum}")
        print(f"  evaluate count:  {pred_sum}")
        self.train_ground_truths = []
        self.train_predictions = []

    def on_validation_epoch_end(self):
        # Aggregate validation set counts
        ground_truths = torch.cat(self.val_ground_truths, dim=0)  # [M, 5]
        predictions = torch.cat(self.val_predictions, dim=0)  # [M, 5]
        gt_sum = ground_truths.sum(dim=0).round().int().numpy()  # [5]
        pred_sum = predictions.sum(dim=0).round().int().numpy()  # [5]
        print(f"\nValidation set --")
        print(f"    ground truth:  {gt_sum}")
        print(f"  evaluate count:  {pred_sum}")
        self.val_ground_truths = []
        self.val_predictions = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=10,
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }  

# Define data module
class SeaLionDataModule(pl.LightningDataModule):
    def __init__(self, train_files, train_dir, dotted_dir, batch_size=8, patch_size=224, num_workers=0, scale_factor=0.4):
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
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, scale=(0.66, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage in (None, "fit"):
            # 創建完整數據集
            full_dataset = SeaLionPatchDataset(
                self.train_files, self.train_dir, self.dotted_dir, self.patch_size, self.scale_factor,
                self.transform, num_workers=self.num_workers
            )
            
            # 計算分割大小
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            # 使用 random_split
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
        )

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train Sea Lion Counting Model")
    parser.add_argument("--train_dir", type=str, default="Train", help="Path to train directory")
    parser.add_argument("--dotted_dir", type=str, default="TrainDotted", help="Path to dotted directory")
    parser.add_argument("--save_model", type=str, default="output", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches to extract from images")
    parser.add_argument("--scale_factor", type=float, default=1, help="Scale factor for image patches")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma value for Focal Loss")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()
    
    # Calculate alpha weights
    print("Calculating class weights...")
    csv_file = os.path.join(args.train_dir, "train.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} not found.")
    train_df = pd.read_csv(csv_file)
    categories = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']
    freq = train_df[categories].mean().values
    alpha = 1 / freq
    alpha = alpha / alpha.sum()  # Normalize
    alpha = torch.tensor(alpha, dtype=torch.float32)
    print(f"Class weights (alpha): {alpha.numpy()}")

    # Prepare data
    print("Preparing dataset...")
    train_files = [f for f in os.listdir(args.train_dir) if f.endswith('.png') or f.endswith('.jpg')]
    data_module = SeaLionDataModule(
        train_files=train_files,
        train_dir=args.train_dir,
        dotted_dir=args.dotted_dir,
        batch_size=4,  # 減小 batch size
        num_workers=4,  # 設置合適的 worker 數
        scale_factor=0.4,
        patch_size=256
    )
    
    print("Initializing model...")
    # Initialize model
    model = SeaLionCountingModel(num_classes=5, learning_rate=args.lr, batch_size=args.batch_size, alpha=alpha, gamma=args.gamma)
    print("Setting up training...")
    # Set up Logger
    logger = TensorBoardLogger("lightning_logs", name="sea_lion_counting")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_model,
        filename="swinv2_pytorch-{epoch:02d}-{val_rmse:.4f}",
        monitor="val_rmse",
        mode="min",
        save_top_k=5,
        save_last=True,
        every_n_epochs=10
    )

    # Train
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