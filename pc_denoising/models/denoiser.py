from typing import Any

import MinkowskiEngine as ME
import torch
from torchmetrics.classification import BinaryJaccardIndex
import pytorch_lightning as pl

from pc_denoising.models.minkunet import MinkUNet34C
from pc_denoising.models.voxelnet import SVFE


class DenseDenoiser(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.svfe = SVFE()
        self.unet = MinkUNet34C(in_channels=128, out_channels=35)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.iou = BinaryJaccardIndex()

    def training_step(self, *args: Any, **kwargs: Any) -> torch.FloatTensor:
        (voxel_coords, voxel_features, voxel_labels), *_ = args
        vwfs = self.svfe(voxel_features)

        inputs = ME.SparseTensor(vwfs, voxel_coords)
        outputs = self.unet(inputs)

        loss = self.loss(outputs.F, voxel_labels.float())
        self.iou(outputs.F, voxel_labels)
        self.log_dict({"train_loss": loss, "train_iou": self.iou}, prog_bar=True)
        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        (voxel_coords, voxel_features, voxel_labels), *_ = args
        vwfs = self.svfe(voxel_features)

        inputs = ME.SparseTensor(vwfs, voxel_coords)
        outputs = self.unet(inputs)

        loss = self.loss(outputs.F, voxel_labels.float())
        self.iou(outputs.F, voxel_labels)
        self.log_dict({"val_loss": loss, "val_iou": self.iou}, prog_bar=True)

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        (voxel_coords, voxel_features, voxel_labels), *_ = args
        vwfs = self.svfe(voxel_features)

        inputs = ME.SparseTensor(vwfs, voxel_coords)
        outputs = self.unet(inputs)

        loss = self.loss(outputs.F, voxel_labels.float())
        self.iou(outputs.F, voxel_labels)
        self.log_dict({"test_loss": loss, "test_iou": self.iou})

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.FloatTensor:
        voxel_coords, voxel_features, voxel_labels = batch
        vwfs = self.svfe(voxel_features)
        inputs = ME.SparseTensor(vwfs, voxel_coords)
        return self.unet(inputs).F
