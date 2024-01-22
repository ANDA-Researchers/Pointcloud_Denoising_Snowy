from __future__ import division

import numpy as np
import torch.utils.data as data

from pc_denoising.voxelnet import utils
from pc_denoising.voxelnet.config import config as cfg
from pc_denoising.voxelnet.data_aug import aug_data


class KittiDataset(data.Dataset):
    def __init__(
        self,
        lidar_files,
        calib_files,
        label_files,
        type="velodyne_train",
    ) -> None:
        self.lidar_files = lidar_files
        self.calib_files = calib_files
        if type == "velodyne_train":
            self.label_files = label_files

        self.type = type

        self.T = cfg.T
        self.vd = cfg.vd
        self.vh = cfg.vh
        self.vw = cfg.vw
        self.xrange = cfg.xrange
        self.yrange = cfg.yrange
        self.zrange = cfg.zrange
        self.anchors = cfg.anchors.reshape(-1, 7)
        self.feature_map_shape = (int(cfg.H / 2), int(cfg.W / 2))
        self.anchors_per_position = cfg.anchors_per_position
        self.pos_threshold = cfg.pos_threshold
        self.neg_threshold = cfg.neg_threshold

    def preprocess(self, lidar):
        # shuffling the points
        np.random.shuffle(lidar)

        voxel_coords = (
            (lidar[:, :3] - np.array([self.xrange[0], self.yrange[0], self.zrange[0]]))
            / (self.vw, self.vh, self.vd)
        ).astype(np.int32)

        # convert to  (D, H, W)
        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(
            voxel_coords, axis=0, return_inverse=True, return_counts=True
        )

        voxel_features = []
        voxel_labels = []

        for i in range(len(voxel_coords)):
            features = np.zeros((self.T, 7), dtype=np.float32)
            labels = np.zeros((self.T), dtype=np.int64)
            pts = lidar[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[: self.T, :]
                voxel_counts[i] = self.T
            # augment the points
            features[: pts.shape[0], :] = np.concatenate(
                (pts[:, :4], pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1
            )
            labels[: pts.shape[0]] = pts[:, 4]
            voxel_features.append(features)
            voxel_labels.append(labels)
        return voxel_coords, np.array(voxel_features), np.array(voxel_labels)

    def __getitem__(self, i):
        lidar_file = self.lidar_files[i]
        calib_file = self.calib_files[i]

        calib = utils.load_kitti_calib(calib_file)
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)

        if self.type == "velodyne_train":
            label_file = self.label_files[i]
            Tr = calib["Tr_velo2cam"]
            gt_box3d = utils.load_kitti_label(label_file, Tr)
            # data augmentation
            lidar, gt_box3d = aug_data(lidar, gt_box3d)
        elif self.type == "velodyne_test":
            pass
        else:
            raise ValueError("the type invalid")

        # specify a range
        lidar, gt_box3d = utils.get_filtered_lidar(lidar, gt_box3d)

        # voxelize
        return self.preprocess(lidar)

    def __len__(self):
        return len(self.lidar_files)
