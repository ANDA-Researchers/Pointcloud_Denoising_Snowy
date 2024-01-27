import itertools
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

    def __init__(self, root: str, subset: str) -> None:
        super().__init__()
        self._root = root
        self._subset = subset

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._train_ds = self._get_train_ds()
        if stage in {"evaluate", "predict"}:
            self._test_ds = self._get_test_ds()
        if stage in {"fit", "validate"}:
            self._val_ds = self._get_val_ds()

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

    def _get_train_ds(self) -> KittiDataset:
        with open(os.path.join(self._root, "kitti_infos_train.pkl"), "rb") as info_ref:
            infos = pickle.load(info_ref)
        id_list = self._get_id_list(infos)
        lidar_files = self._get_fit_lidar_files(self._subset, id_list)
        calib_files = [
            os.path.join(self._root, self._subset, "calib", f"{file}.txt")
            for file in id_list
        ]
        label_files = [
            os.path.join(self._root, self._subset, "label_2", f"{file}.txt")
            for file in id_list
        ]
        return KittiDataset(lidar_files, calib_files, label_files, "velodyne_train")

    def _get_test_ds(self) -> KittiDataset:
        with open(os.path.join(self._root, "kitti_infos_test.pkl"), "rb") as info_ref:
            infos = pickle.load(info_ref)
        id_list = self._get_id_list(infos)
        lidar_files = self._get_test_lidar_files(id_list)
        calib_files = [
            [
                os.path.join(self._root, "testing", "calib", f"{file}.txt")
                for file in id_list
            ]
            for _ in range(3)  # Replicate 3 times for light, medium and heavy
        ]
        calib_files = list(itertools.chain.from_iterable(calib_files))
        return KittiDataset(lidar_files, calib_files, None, "velodyne_test")

    def _get_val_ds(self) -> KittiDataset:
        with open(os.path.join(self._root, "kitti_infos_val.pkl"), "rb") as info_ref:
            infos = pickle.load(info_ref)
        id_list = self._get_id_list(infos)
        lidar_files = self._get_fit_lidar_files(self._subset, id_list)
        calib_files = [
            os.path.join(self._root, self._subset, "calib", f"{file}.txt")
            for file in id_list
        ]
        return KittiDataset(lidar_files, calib_files, None, "velodyne_test")

    def _get_id_list(self, infos: Sequence[Mapping[str, Any]]) -> List[str]:
        return [info["velodyne_path"].split("/")[-1].split(".")[0] for info in infos]

    def _get_fit_lidar_files(self, subset: int, id_list: Sequence[str]) -> List[str]:
        if subset == 0:
            return self._get_all_lidar_files(id_list)
        elif 0 < subset < 4:
            return self._get_mixed_lidar_files(subset, id_list)
        else:
            return self._get_lidar_files(subset, id_list)

    def _get_test_lidar_files(self, id_list: Sequence[str]):
        lidar_file = os.path.join(
            self._root,
            "testing",
            "velodyne_reduced",
            "{subset}",
            "{file}.bin",
        )
        lidar_files = []
        lidar_files.extend([lidar_file.format("light", id_) for id_ in id_list])
        lidar_files.extend([lidar_file.format("medium", id_) for id_ in id_list])
        lidar_files.extend([lidar_file.format("heavy", id_) for id_ in id_list])
        return lidar_files

    def _get_lidar_files(self, subset: int, id_list: Sequence[str]) -> List[str]:
        return [
            os.path.join(
                self._root,
                "training",
                "velodyne_reduced",
                self._subsets[subset],
                f"{id_}.bin",
            )
            for id_ in id_list
        ]

    def _get_mixed_lidar_files(self, subset: int, id_list: Sequence[str]):
        subsets = self._mixed_subsets[subset]
        lidar_files = []
        lidar_file = os.path.join(
            self._root,
            "training",
            "velodyne_reduced",
            "{subset}",
            "{file}.bin",
        )
        for idx, id_ in enumerate(id_list):
            if idx % 2 == 0:
                lidar_files.append(lidar_file.format(subsets[0], id_))
            else:
                lidar_files.append(lidar_file.format(subsets[1], id_))
        return lidar_files

    def _get_all_lidar_files(self, id_list: Sequence[str]):
        lidar_files = []
        lidar_file = os.path.join(
            self._root,
            "training",
            "velodyne_reduced",
            "{subset}",
            "{file}",
        )
        for idx, id_ in enumerate(id_list):
            if idx % 3 == 0:
                lidar_files.append(lidar_file.format("light", id_))
            elif idx % 3 == 1:
                lidar_files.append(lidar_file.format("medium", id_))
            else:
                lidar_files.append(lidar_file.format("heavy", id_))
        return lidar_files
