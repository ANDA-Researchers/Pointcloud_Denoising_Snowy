from typing import Any

import MinkowskiEngine as ME
import torch
from torchmetrics.classification import BinaryJaccardIndex
import pytorch_lightning as pl

from pc_denoising.minkunet import MinkUNet34C
from pc_denoising.voxelnet import voxelnet


class MinkowskiUNet(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.svfe = voxelnet.SVFE()
        self.unet = MinkUNet34C(in_channels=128, out_channels=35)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.train_iou = BinaryJaccardIndex()
        self.test_iou = BinaryJaccardIndex()
        self.val_iou = BinaryJaccardIndex()

    def training_step(self, *args: Any, **kwargs: Any) -> torch.FloatTensor:
        (voxel_coords, voxel_features, voxel_labels), *_ = args
        vwfs = self.svfe(voxel_features)

        inputs = ME.SparseTensor(vwfs, voxel_coords)
        outputs = self.unet(inputs)

        loss = self.loss(outputs.F, voxel_labels.float())
        self.train_iou(outputs.F, voxel_labels)
        self.log_dict({"iou": self.train_iou}, prog_bar=True)
        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        (voxel_coords, voxel_features, voxel_labels), *_ = args
        vwfs = self.svfe(voxel_features)

        inputs = ME.SparseTensor(vwfs, voxel_coords)
        outputs = self.unet(inputs)

        loss = self.loss(outputs.F, voxel_labels.float())
        self.val_iou(outputs.F, voxel_labels)
        self.log_dict({"val_loss": loss, "val_iou": self.val_iou}, prog_bar=True)

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        (voxel_coords, voxel_features, voxel_labels), *_ = args
        vwfs = self.svfe(voxel_features)

        inputs = ME.SparseTensor(vwfs, voxel_coords)
        outputs = self.unet(inputs)

        loss = self.loss(outputs.F, voxel_labels.float())
        self.test_iou(outputs.F, voxel_labels)
        self.log_dict({"test_loss": loss, "test_iou": self.test_iou}, prog_bar=True)
