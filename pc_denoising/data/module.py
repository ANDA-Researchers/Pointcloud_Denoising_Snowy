import os
import pickle
from multiprocessing import cpu_count
from typing import Any, List, Mapping, Sequence, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pc_denoising.config import config as cfg
from pc_denoising.data.dataset import KittiDataset
from pc_denoising.data.utils import segmentation_collate


class KITTI(pl.LightningDataModule):
    _subsets: Mapping[int, str] = {
        4: "heavy",
        5: "medium",
        6: "light",
    }
    _mixed_subsets: Mapping[int, Tuple[str, str]] = {
        1: ("light", "medium"),
        2: ("light", "heavy"),
        3: ("medium", "heavy"),
    }

    def __init__(self, root: str, subset: int) -> None:
        super().__init__()
        self._root = root
        self._subset = subset

    def setup(self, stage: str) -> None:
        if stage == "fit":
            with open(
                os.path.join(self._root, "kitti_infos_train.pkl"), "rb"
            ) as info_ref:
                infos = pickle.load(info_ref)
            self._train_ds = self._get_ds(infos, "training")
        if stage in {"test", "predict"}:
            with open(
                os.path.join(self._root, "kitti_infos_test.pkl"), "rb"
            ) as info_ref:
                infos = pickle.load(info_ref)
            self._test_ds = self._get_ds(infos, "testing", "velodyne_test")
        if stage in {"fit", "validate"}:
            with open(
                os.path.join(self._root, "kitti_infos_val.pkl"), "rb"
            ) as info_ref:
                infos = pickle.load(info_ref)
            self._val_ds = self._get_ds(infos, "training", "velodyne_test")

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
        self,
        infos: Sequence[Mapping[str, Any]],
        stage: str,
        type_: str = "velodyne_train",
    ) -> KittiDataset:
        id_list = self._get_id_list(infos)
        lidar_files = self._dispatch_lidar_files(stage, id_list)
        calib_files = [
            os.path.join(self._root, stage, "calib", f"{file}.txt") for file in id_list
        ]
        if stage == "traning":
            label_files = [
                os.path.join(self._root, "training", "label_2", f"{file}.txt")
                for file in id_list
            ]
        else:
            label_files = None
        return KittiDataset(
            lidar_files,
            calib_files,
            label_files,
            type_,
        )

    def _get_id_list(self, infos: Sequence[Mapping[str, Any]]) -> List[str]:
        return [info["velodyne_path"].split("/")[-1].split(".")[0] for info in infos]

    def _dispatch_lidar_files(self, stage: str, id_list: Sequence[str]) -> List[str]:
        if self._subset == 0:
            return self._get_all_lidar_files(stage, id_list)
        if 0 < self._subset < 4:
            return self._get_mixed_lidar_files(stage, id_list)
        return self._get_lidar_files(stage, id_list)

    def _get_lidar_files(self, stage: str, id_list: Sequence[str]) -> List[str]:
        return [
            os.path.join(
                self._root,
                stage,
                "velodyne_reduced",
                self._subsets[self._subset],
                f"{id_}.bin",
            )
            for id_ in id_list
        ]

    def _get_mixed_lidar_files(self, stage: str, id_list: Sequence[str]):
        subsets = self._mixed_subsets[self._subset]
        lidar_files = []
        lidar_file = os.path.join(
            self._root,
            stage,
            "velodyne_reduced",
            "{}",
            "{}.bin",
        )
        for idx, id_ in enumerate(id_list):
            if idx % 2 == 0:
                lidar_files.append(lidar_file.format(subsets[0], id_))
            else:
                lidar_files.append(lidar_file.format(subsets[1], id_))
        return lidar_files

    def _get_all_lidar_files(self, stage: str, id_list: Sequence[str]):
        lidar_files = []
        lidar_file = os.path.join(
            self._root,
            stage,
            "velodyne_reduced",
            "{}",
            "{}.bin",
        )
        for idx, id_ in enumerate(id_list):
            if idx % 3 == 0:
                lidar_files.append(lidar_file.format("light", id_))
            elif idx % 3 == 1:
                lidar_files.append(lidar_file.format("medium", id_))
            else:
                lidar_files.append(lidar_file.format("heavy", id_))
        return lidar_files
