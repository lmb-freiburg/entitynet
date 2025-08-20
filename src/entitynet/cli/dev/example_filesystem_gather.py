"""
Test script for filesystem-based gathering functionality in distributed training.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import lightning as lit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from visiontext.distutils import WorldInfo

from entitynet.litext.distributed_gathering import save_outputs

TEMP_DIR = "/ihome/gings/repos/workspace/clip_project/TEMP_ALLGATHER"


class DummyDataset(Dataset):
    """Simple dataset that returns random data with indices."""

    def __init__(self, size: int = 100, input_dim: int = 32, num_classes: int = 10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.randn(self.input_dim),
            "y": torch.randint(0, self.num_classes, (1,)).squeeze(),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


class DummyModel(lit.LightningModule):
    """Simple neural network for testing distributed gathering."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.validation_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch["x"], batch["y"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y, idx = batch["x"], batch["y"], batch["idx"]
        keys = [f"key_{i}" for i in idx]
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Store outputs for gathering
        output = {
            "logits": logits,
            "targets": y,
            "idx": idx,
            "keys": keys,
        }
        self.validation_outputs.append(output)
        self.log("val_loss", loss)
        return output

    def on_validation_epoch_end(self) -> None:
        """Test filesystem gathering using the save_outputs function."""
        if len(self.validation_outputs) == 0:
            raise ValueError("No validation outputs to gather")

        wi = WorldInfo(self.trainer)

        # Ensure temp directory exists and is clean
        if wi.is_global_zero:
            os.makedirs(TEMP_DIR, exist_ok=True)
            # Clean any existing files from previous runs
            for file in Path(TEMP_DIR).glob("*"):
                if file.is_file():
                    file.unlink(missing_ok=True)
        wi.barrier_safe()  # Wait for rank 0 to clean up

        print(
            f"Rank {wi.global_rank}: Testing save_outputs with {len(self.validation_outputs)} batches"
        )

        # Create target file path
        target_file = Path(TEMP_DIR) / "test_outputs.pt"

        # Call save_outputs with the required parameters
        clean_outputs = save_outputs(
            trainer=self.trainer,
            test_outputs=self.validation_outputs,
            all_gather_fn=self.all_gather,
            target_file=target_file,
            target_file_extras=None,  # Leave out optional parameters
            skip_save=True,  # Don't actually save files, just test gathering
            verbose=True,
        )

        # Only rank 0 gets the clean outputs
        if wi.is_global_zero and clean_outputs is not None:
            print(f"Rank 0: Successfully gathered and cleaned outputs")
            print(f"Final output keys: {list(clean_outputs.keys())}")

            # Print some info about the outputs
            for key, value in clean_outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: tensor shape {value.shape}")
                elif isinstance(value, list):
                    print(f"  - {key}: list length {len(value)}")
                else:
                    print(f"  - {key}: {type(value)}")

            print("save_outputs test passed!")

        # Clear validation outputs for next epoch
        self.validation_outputs.clear()

    def _consolidate_local_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate outputs from this process."""
        if not outputs:
            return {}

        consolidated = {}
        for key in outputs[0].keys():
            values = [out[key] for out in outputs]
            print(f"Consolidationg {key} with {len(values)} values")

            if isinstance(values[0], torch.Tensor):
                consolidated[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], (int, float)):
                consolidated[key] = values  # Keep as list
            else:
                consolidated[key] = values

        return consolidated

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def create_dataloaders(batch_size: int = 16, num_workers: int = 2) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = DummyDataset(size=200)
    val_dataset = DummyDataset(size=100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


def main():
    """Main function to test filesystem gathering."""
    parser = argparse.ArgumentParser(description="Test filesystem gathering with PyTorch Lightning")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")

    args = parser.parse_args()

    print(f"Starting filesystem gathering test with {args.devices} device(s)")

    # Create model and data
    model = DummyModel()
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Setup trainer
    trainer = lit.Trainer(
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() and args.devices > 0 else "cpu",
        strategy="ddp" if args.devices > 1 else "auto",
        max_epochs=args.max_epochs,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print(f"Trainer setup: {trainer.num_devices} devices, strategy: {trainer.strategy}")

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    print("Filesystem gathering test completed successfully!")


if __name__ == "__main__":
    main()
