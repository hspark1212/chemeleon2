"""Lightning DataModule for crystal structure datasets.

This module provides a LightningDataModule wrapper for loading and batching
crystal structure data with support for Materials Project datasets.
"""

from lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from src.data.components.mp_dataset import MPDataset
from src.data.schema import CrystalBatch


class DataModule(LightningDataModule):
    """PyTorch Lightning DataModule for crystal structure datasets."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        dataset_type: str = "mp",
        target_condition: str | None = None,
        mace_features: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        # Configs for dataset
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.target_condition = target_condition
        self.mace_features = mace_features
        print(f"Data directory: {self.data_dir}")

        # Configs for dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def dataset_cls(self):
        """Return the dataset class based on the dataset type."""
        return MPDataset

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_cls(
                root=self.data_dir,
                split="train",
                target_condition=self.target_condition,
                mace_features=self.mace_features,
            )
            self.val_dataset = self.dataset_cls(
                root=self.data_dir,
                split="val",
                target_condition=self.target_condition,
                mace_features=self.mace_features,
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_cls(
                root=self.data_dir,
                split="test",
                target_condition=self.target_condition,
                mace_features=self.mace_features,
            )

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        loader.collate_fn = CrystalBatch.collate
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        loader.collate_fn = CrystalBatch.collate
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        loader.collate_fn = CrystalBatch.collate
        return loader
