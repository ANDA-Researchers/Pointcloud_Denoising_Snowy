from __future__ import division

import os
import os.path

import cv2
import numpy as np
import torch.utils.data as data

import pc_denoising.voxel_net.config as cfg
from pc_denoising.voxel_net import utils
from pc_denoising.voxel_net.data_aug import aug_data


class KittiDataset(data.Dataset):
    def __init__(
        self, lidar_path, image_path, calib_path, label_path, type="velodyne_train"
    ):
        self.type = type
        self.lidar_path = lidar_path
        self.image_path = image_path
        self.calib_path = calib_path
        self.label_path = label_path

        with open(os.path.join(self.data_path, "%s.txt" % set)) as f:
            self.file_list = f.read().splitlines()

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

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 7), dtype=np.float32)
            pts = lidar[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[: self.T, :]
                voxel_counts[i] = self.T
            # augment the points
            voxel[: pts.shape[0], :] = np.concatenate(
                (pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1
            )
            voxel_features.append(voxel)
        return np.array(voxel_features), voxel_coords

    def __getitem__(self, i):
        lidar_file = self.lidar_path + "/" + self.file_list[i] + ".bin"
        calib_file = self.calib_path + "/" + self.file_list[i] + ".txt"
        label_file = self.label_path + "/" + self.file_list[i] + ".txt"
        image_file = self.image_path + "/" + self.file_list[i] + ".png"

        calib = utils.load_kitti_calib(calib_file)
        Tr = calib["Tr_velo2cam"]
        gt_box3d = utils.load_kitti_label(label_file, Tr)
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

        image = cv2.imread(image_file)

        # data augmentation
        if self.type == "velodyne_train":
            lidar, gt_box3d = aug_data(lidar, gt_box3d)

        # specify a range
        lidar, gt_box3d = utils.get_filtered_lidar(lidar, gt_box3d)

        # voxelize
        voxel_features, voxel_coords = self.preprocess(lidar)

        return (
            voxel_features,
            voxel_coords,
            image,
            calib,
            self.file_list[i],
        )

    def __len__(self):
        return len(self.file_list)
