from __future__ import division

import numpy as np
import torch

from pc_denoising.config import config as cfg


def get_filtered_lidar(lidar, boxes3d=None):
    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]

    filter_x = np.where((pxs >= cfg.xrange[0]) & (pxs < cfg.xrange[1]))[0]
    filter_y = np.where((pys >= cfg.yrange[0]) & (pys < cfg.yrange[1]))[0]
    filter_z = np.where((pzs >= cfg.zrange[0]) & (pzs < cfg.zrange[1]))[0]
    filter_xy = np.intersect1d(filter_x, filter_y)
    filter_xyz = np.intersect1d(filter_xy, filter_z)

    if boxes3d is not None:
        box_x = (boxes3d[:, :, 0] >= cfg.xrange[0]) & (boxes3d[:, :, 0] < cfg.xrange[1])
        box_y = (boxes3d[:, :, 1] >= cfg.yrange[0]) & (boxes3d[:, :, 1] < cfg.yrange[1])
        box_z = (boxes3d[:, :, 2] >= cfg.zrange[0]) & (boxes3d[:, :, 2] < cfg.zrange[1])
        box_xyz = np.sum(box_x & box_y & box_z, axis=1)

        return lidar[filter_xyz], boxes3d[box_xyz > 0]

    return lidar[filter_xyz], boxes3d


def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert len(lines) == 8

    obj = lines[0].strip().split(" ")[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(" ")[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(" ")[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(" ")[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(" ")[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(" ")[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(" ")[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {
        "P2": P2.reshape(3, 4),
        "R0": R0.reshape(3, 3),
        "Tr_velo2cam": Tr_velo_to_cam.reshape(3, 4),
    }


def box3d_cam_to_velo(box3d, Tr):
    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2 * np.pi + angle

        return angle

    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )

    rz = ry_to_rz(ry)

    rotMat = np.array(
        [[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]]
    )

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()

    return box3d_corner.astype(np.float32)


def load_kitti_label(label_file, Tr):
    with open(label_file, "r") as f:
        lines = f.readlines()

    gt_boxes3d_corner = []

    num_obj = len(lines)

    for j in range(num_obj):
        obj = lines[j].strip().split(" ")

        obj_class = obj[0].strip()
        if obj_class not in cfg.class_list:
            continue

        box3d_corner = box3d_cam_to_velo(obj[8:], Tr)

        gt_boxes3d_corner.append(box3d_corner)

    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1, 8, 3)

    return gt_boxes3d_corner


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
