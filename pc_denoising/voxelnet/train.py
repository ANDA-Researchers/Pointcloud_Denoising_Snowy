import torch
import numpy as np


def segmentation_collate(batch):
    voxel_coords = []
    voxel_features = []
    voxel_labels = []

    for i, sample in enumerate(batch):
        voxel_coords.append(
            np.pad(sample[0], ((0, 0), (1, 0)), mode="constant", constant_values=i)
        )
        voxel_features.append(sample[1])
        voxel_labels.append(sample[2])
    return (
        torch.from_numpy(np.concatenate(voxel_coords)),
        torch.from_numpy(np.concatenate(voxel_features)),
        torch.from_numpy(np.concatenate(voxel_labels)),
    )
