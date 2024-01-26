import os
import pickle
from multiprocessing import cpu_count
from typing import Any, Mapping, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pc_denoising.config import config as cfg
from pc_denoising.data.dataset import KittiDataset
from pc_denoising.data.utils import segmentation_collate


class KITTI(pl.LightningDataModule):
    def __init__(self, root: str, snow_rate: str) -> None:
        super().__init__()
        self._root = root
        self._train_info_file = os.path.join(root, "kitti_infos_train.pkl")
        self._test_info_file = os.path.join(root, "kitti_infos_test.pkl")
        self._val_info_file = os.path.join(root, "kitti_infos_val.pkl")
        self._snow_rate = snow_rate

    def setup(self, stage: str) -> None:
        if stage == "fit":
            with open(self._train_info_file, "rb") as info_ref:
                infos = pickle.load(info_ref)
                self._train_ds = self._get_ds(infos, "training", "velodyne_train")
        if stage in {"fit", "validate"}:
            with open(self._val_info_file, "rb") as info_ref:
                infos = pickle.load(info_ref)
                self._val_ds = self._get_ds(infos, "training", "velodyne_test")
        if stage in {"evaluate", "predict"}:
            with open(self._test_info_file, "rb") as info_ref:
                infos = pickle.load(info_ref)
                self._test_ds = self._get_ds(infos, "testing", "velodyne_test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            cfg.N,
            shuffle=True,
            collate_fn=segmentation_collate,
            num_workers=cpu_count(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            cfg.N,
            collate_fn=segmentation_collate,
            num_workers=cpu_count(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            cfg.N,
            collate_fn=segmentation_collate,
            num_workers=cpu_count(),
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def _get_ds(
        self, infos: Sequence[Mapping[str, Any]], subset: str, type_: str
    ) -> KittiDataset:
        file_list = [
            info["velodyne_path"].split("/")[-1].split(".")[0] for info in infos
        ]
        lidar_files = [
            os.path.join(
                self._root,
                "training",
                "velodyne_reduced",
                self._snow_rate,
                f"{file}.bin",
            )
            for file in file_list
        ]
        calib_files = [
            os.path.join(self._root, subset, "calib", f"{file}.txt")
            for file in file_list
        ]
        label_files = [
            os.path.join(self._root, subset, "label_2", f"{file}.txt")
            for file in file_list
        ]
        return KittiDataset(lidar_files, calib_files, label_files, type_)
