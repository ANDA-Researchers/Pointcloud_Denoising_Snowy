import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


class KITTI(Dataset):
    _range = np.array([[0.0, 70.4], [-40.0, 40.0], [-3.0, 1.0]])
    _scale = np.array([0.2, 0.2, 0.4])

    def __init__(
        self, root_dir: Path, info_path: Path, is_reduced: bool = True
    ) -> None:
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
        lidar = _load_lidar(self._lidar_paths[idx])
        voxel_coords = _voxel_partition(lidar[:, :3], self._range[:, 0], self._scale)
        return voxel_coords


def _load_lidar(lidar_path: Path) -> npt.NDArray[np.float32]:
    return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)


def _voxel_partition(
    voxel_coords: npt.NDArray[np.float32],
    offset: npt.NDArray[np.float32],
    scale: npt.NDArray[np.float32],
) -> npt.NDArray[np.int32]:
    return ((voxel_coords - offset) / scale)[:, [2, 0, 1]].astype(np.int32)
