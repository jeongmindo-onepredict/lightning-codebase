import os
import sys
import glob
import torch
import numpy as np
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import lightning as L


class FaultDataset(Dataset):
    def __init__(
            self, 
            # transform: Any,
            paths: list,
        ):
        self.paths = paths
        # self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        data = np.load(self.paths[idx])
        label = self.paths[idx].split("/")[-2] # Fault/6.npy 인 경우 Fault를 뽑아냄
        motor_id = self.paths[idx].split("/")[-1].split(".")[0] # Fault/6.npy인 경우 6을 뽑아냄
        # data = self.transform(data)
        
        # to tensor
        data = torch.tensor(data)
        label = torch.tensor(label)
        motor_id = torch.tensor(motor_id)

        return data, label, motor_id



class ClassificationLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        # transforms: Any, # augmentation을 위한 transforms 현재는 없음
        num_workers: int,
    ) -> None:
        super().__init__()
        self.root = root
        # self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        """
        정형 데이터는 여기서 불러오고, 비정형 데이터는 하위 torch의 dataset을 상속받은 곳에서 불러옵니다. (한 번에 다 불러오면 메모리 터져요... 위치만 담고, 배열은 배치로 불러오는 패턴을 사용)

        ex) 
        1. numpy 배열을 학습에 사용할 경우 .npy의 위치 및 메타정보를 담은 파일(e.g. csv)를 불러오는 작업을 여기서 수행합니다.
        2. torch의 dataset을 상속받은 곳에서 .npy를 불러오는 작업을 수행합니다.
        """
        train_paths = len(glob.glob(os.path.join(self.root, "train") + "/*/*.npy"))
        test_paths = len(glob.glob(os.path.join(self.root, "test") + "/*/*.npy"))

        train_paths, valid_paths = train_test_split(
            train_paths, test_size=0.2, random_state=42
        )

        self.train_dataset = FaultDataset(train_paths)
        self.val_dataset = FaultDataset(valid_paths)
        self.test_dataset = FaultDataset(test_paths)

    def _collate_fn(self, batch):
        # padding과 같은 데이터를 불러올 때의 처리를 정의
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
    