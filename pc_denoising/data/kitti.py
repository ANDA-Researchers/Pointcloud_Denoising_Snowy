import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


class KITTI(Dataset):
    _lidar_range = np.array([[0.0, 70.4], [-40.0, 40.0], [-3.0, 1.0]], dtype=np.float32)
    _voxel_size = np.array([0.2, 0.2, 0.4], dtype=np.float32)
    _num_points_per_voxel = 35

    def __init__(self, lidar_paths: Sequence[Path]) -> None:
        self._lidar_paths = lidar_paths

    def __len__(self) -> int:
        return len(self._lidar_paths)

    def __getitem__(self, idx: int):
        lidar = _load_lidar(self._lidar_paths[idx])
        voxel_coords, voxel_indices, point_counts = self._partition_voxels(lidar[:, :3])
        sampled_points = self._sample_points(
            lidar, voxel_coords, voxel_indices, point_counts
        )
        voxel_features = sampled_points[:, :, :4]
        voxel_labels = sampled_points[:, :, 4]
        voxel_features = _augment_features(voxel_features)
        return voxel_coords, voxel_features, voxel_labels

    def _partition_voxels(
        self, voxel_coords: npt.NDArray[np.float32]
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        voxel_coords = (voxel_coords - self._lidar_range[:, 0]) / self._voxel_size
        voxel_coords = voxel_coords[:, [2, 0, 1]].astype(np.int64)
        return np.unique(voxel_coords, return_inverse=True, return_counts=True, axis=0)

    def _sample_points(
        self,
        lidar: npt.NDArray[np.float32],
        voxel_coords: npt.NDArray[np.int64],
        voxel_indices: npt.NDArray[np.int64],
        point_counts: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        num_voxels = len(voxel_coords)
        sampled_points = np.zeros(
            (num_voxels, self._num_points_per_voxel, 5), dtype=np.float32
        )
        for idx, point_count in enumerate(point_counts):
            points_in_voxel = lidar[voxel_indices == idx]
            num_sampled_points = min(self._num_points_per_voxel, point_count)
            sampled_points[idx, :num_sampled_points] = random.sample(
                points_in_voxel.tolist(), num_sampled_points
            )
        return sampled_points


def _load_lidar(lidar_path: Path) -> npt.NDArray[np.float32]:
    return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)


def _augment_features(
    voxel_features: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    voxel_centroids = np.mean(voxel_features[:, :, :3], axis=1)
    augmented_features = voxel_features[:, :, :3] - np.expand_dims(
        voxel_centroids, axis=1
    )
    return np.concatenate([voxel_features, augmented_features], axis=2)
