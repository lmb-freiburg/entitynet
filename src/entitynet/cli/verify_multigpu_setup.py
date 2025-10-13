"""
Basic example to check if multi-GPU setup is working correctly with Lightning.
"""
import argparse
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class DummyDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, num_samples=1000):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
    
    def setup(self, stage=None):
        X = torch.randn(self.num_samples, 10)
        y = torch.randn(self.num_samples, 1)
        self.dataset = TensorDataset(X, y)
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)


def verify_multigpu_setup(num_gpus):
    print(f"PyTorch version: {torch.__version__}")
    print(f"Lightning version: {L.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # Print GPU names
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create model and data
    model = SimpleModel()
    datamodule = DummyDataModule(batch_size=32)
    
    # Configure trainer for specified number of GPUs
    trainer = L.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=2,
        log_every_n_steps=10,
        enable_checkpointing=False,
        logger=True
    )
    
    print("\n" + "="*50)
    print(f"Starting multi-GPU training test with {num_gpus} GPUs...")
    print("="*50 + "\n")
    
    # Run training
    trainer.fit(model, datamodule)
    
    print("\n" + "="*50)
    print("Multi-GPU setup verification complete!")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify multi-GPU setup with Lightning")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use (default: 8)")
    args = parser.parse_args()
    
    verify_multigpu_setup(args.num_gpus)
