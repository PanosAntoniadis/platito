import lightning as L
from typing import Optional, Callable
from torch.utils.data import Dataset, DataLoader

from platito.data import MDCATH


class MDCATHDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        protein_names: list,
        seq_emb_name: str,
        temperatures: list,
        max_lag: int,
        samples_per_epoch: int,
        collate_fn: Callable,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset: Optional[Dataset] = None
        self.dataset_path = dataset_path
        self.protein_names = protein_names
        self.seq_emb_name = seq_emb_name
        self.temperatures = temperatures
        self.max_lag = max_lag
        self.samples_per_epoch = samples_per_epoch
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = MDCATH(
                dataset_path=self.dataset_path,
                protein_names=self.protein_names,
                seq_emb_name=self.seq_emb_name,
                temperatures=self.temperatures,
                max_lag=self.max_lag,
                samples_per_epoch=self.samples_per_epoch,
            )
            print(f"Train set size: {len(self.train_dataset)}")

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )
        return dataloader
