import pickle
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


class KITTI(Dataset):
    _lidar_range = np.array([[0.0, 70.4], [-40.0, 40.0], [-3.0, 1.0]])
    _voxel_size = np.array([0.2, 0.2, 0.4])
    _num_points_per_voxel = 35

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
        np.random.shuffle(lidar)
        voxel_coords, voxel_indices = self._partition_voxels(lidar[:, :3])
        point_groups = _group_points(lidar, voxel_coords, voxel_indices)
        sampled_points = self._sample_points(point_groups)
        voxel_features = sampled_points[:, :, 3]
        return voxel_coords, voxel_features

    def _partition_voxels(
        self, voxel_coords: npt.NDArray[np.float32]
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        voxel_coords = (voxel_coords - self._lidar_range[:, 0]) // self._voxel_size
        return np.unique(voxel_coords[:, [2, 0, 1]], return_inverse=True, axis=0)

    def _sample_points(
        self,
        lidar: npt.NDArray[np.float32],
        voxel_coords: npt.NDArray[np.int64],
        voxel_indices: npt.NDArray[np.int64],
    ):
        num_voxels = len(voxel_coords)
        sampled_points = np.zeros(
            (num_voxels, self._num_points_per_voxel, 7), dtype=np.float32
        )
        for idx in range(num_voxels):
            points_in_voxel = lidar[voxel_indices == idx]
            num_sampled_points = min(self._num_points_per_voxel, len(points_in_voxel))
            sampled_points[idx, :num_sampled_points]


def _load_lidar(lidar_path: Path) -> npt.NDArray[np.float32]:
    return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)


def _group_points(
    lidar: npt.NDArray[np.float32],
    voxel_coords: npt.NDArray[np.int64],
    voxel_indices: npt.NDArray[np.int64],
) -> List[npt.NDArray[np.float32]]:
    return [lidar[voxel_indices == idx] for idx, _ in enumerate(voxel_coords)]
