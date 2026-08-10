import json
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datamodules.benchmark_dataset import (
    BenchmarkPatients,
    BenchmarkTrainSlices,
)


class BenchmarkDataModule(LightningDataModule):
    def __init__(self, cfg, fold=None):
        super().__init__()
        self.cfg = cfg
        self.fold = fold

    def _load_fold_patient_ids(self):
        if self.fold is None:
            raise ValueError(
                "Benchmark 5-fold training requires fold=0,1,2,3,or 4"
            )

        split_path = Path(self.cfg.split_file).expanduser()
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r", encoding="utf-8") as file:
            split_definition = json.load(file)

        fold_number = int(self.fold) + 1
        fold_info = next(
            (
                item
                for item in split_definition["folds"]
                if int(item["fold"]) == fold_number
            ),
            None,
        )

        if fold_info is None:
            raise ValueError(
                f"Fold {fold_number} is absent from {split_path}"
            )

        train_ids = fold_info["train_patient_ids"]
        val_ids = fold_info["val_patient_ids"]

        if len(train_ids) != 336 or len(val_ids) != 84:
            raise ValueError(
                f"Fold {fold_number}: expected train=336 and val=84, "
                f"got train={len(train_ids)}, val={len(val_ids)}"
            )

        if set(train_ids) & set(val_ids):
            raise ValueError(f"Fold {fold_number}: train/val overlap")

        return train_ids, val_ids

    def setup(self, stage=None):
        stage_name = "" if stage is None else str(stage).lower()

        is_fit = (
            stage is None
            or stage_name.endswith("fit")
            or stage_name.endswith("fitting")
        )
        is_test_or_predict = (
            stage is None
            or stage_name.endswith("test")
            or stage_name.endswith("testing")
            or stage_name.endswith("predict")
            or stage_name.endswith("predicting")
        )

        if is_fit:
            train_ids, val_ids = self._load_fold_patient_ids()

            self.train_set = BenchmarkTrainSlices(
                self.cfg.train_dir,
                horizontal_flip_p=self.cfg.train_augmentation.horizontal_flip_p,
                patient_ids=train_ids,
            )

            self.val_set = BenchmarkTrainSlices(
                self.cfg.train_dir,
                horizontal_flip_p=0.0,
                patient_ids=val_ids,
            )

            expected_train_slices = 336 * 50
            expected_val_slices = 84 * 50

            if len(self.train_set) != expected_train_slices:
                raise RuntimeError(
                    f"Fold {self.fold}: train has {len(self.train_set)} "
                    f"slices, expected {expected_train_slices}"
                )

            if len(self.val_set) != expected_val_slices:
                raise RuntimeError(
                    f"Fold {self.fold}: val has {len(self.val_set)} "
                    f"slices, expected {expected_val_slices}"
                )

        if is_test_or_predict:
            self.ixi_test_set = BenchmarkPatients(
                self.cfg.ixi_test_dir,
                require_lesion=False,
            )
            self.brats_test_set = BenchmarkPatients(
                self.cfg.brats_test_dir,
                require_lesion=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def ixi_test_dataloader(self):
        return DataLoader(
            self.ixi_test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def brats_test_dataloader(self):
        return DataLoader(
            self.brats_test_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )