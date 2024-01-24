from typing import Tuple

import MinkowskiEngine as ME
import torch
import pytorch_lightning as pl

from pc_denoising.minkunet import MinkUNet34C
from pc_denoising.voxelnet import voxelnet


class MinkowskiUNet(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.svfe = voxelnet.SVFE()
        self.unet = MinkUNet34C(in_channels=128, out_channels=35)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def training_step(
        self, batch: Tuple[torch.IntTensor, torch.FloatTensor, torch.LongTensor]
    ) -> torch.FloatTensor:
        voxel_coords, voxel_features, voxel_labels = batch
        vwfs = self.svfe(voxel_features)

        inputs = ME.SparseTensor(vwfs, voxel_coords)
        outputs = self.unet(inputs)

        loss = self.loss(outputs, voxel_labels)
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.IntTensor, torch.FloatTensor, torch.LongTensor]
    ) -> None:
        voxel_coords, voxel_features, voxel_labels = batch
        vwfs = self.svfe(voxel_features)

        inputs = ME.SparseTensor(vwfs, voxel_coords)
        outputs = self.unet(inputs)

        loss = self.loss(outputs, voxel_labels)
        self.log_dict({"val_loss": loss}, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.IntTensor, torch.FloatTensor, torch.LongTensor]
    ) -> None:
        voxel_coords, voxel_features, voxel_labels = batch
        vwfs = self.svfe(voxel_features)

        inputs = ME.SparseTensor(vwfs, voxel_coords)
        outputs = self.unet(inputs)

        loss = self.loss(outputs, voxel_labels)
        self.log_dict({"test_loss": loss}, prog_bar=True)
