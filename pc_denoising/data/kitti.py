import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


class KITTI(Dataset):
    def __init__(self, root_dir: Path, info_path: Path, is_reduced: bool = True) -> None:
        super().__init__()
        infos = pickle.loads(info_path.read_bytes())
        lidar_dir_name = "velodyne_reduced" if is_reduced else "velodyne"
        self._lidar_paths = [
            root_dir / info["velodyne_path"].replace("velodyne", lidar_dir_name)
            for info in infos
        ]

    def __len__(self) -> int:
        return len(self._lidar_paths)

    def __getitem__(self, idx: int):
        return _load_lidar(self._lidar_paths[idx])


def _load_lidar(lidar_path: Path) -> npt.NDArray[np.float32]:
    return np.fromfile(lidar_path, dtype=np.float32).astype(np.float32).reshape(-1, 5)
