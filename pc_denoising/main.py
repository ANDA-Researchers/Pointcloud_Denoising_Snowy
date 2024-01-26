from pytorch_lightning.cli import LightningCLI

from pc_denoising.data import KITTI
from pc_denoising.models import MinkowskiUNet


def cli_main():
    cli = LightningCLI(MinkowskiUNet, KITTI)


if __name__ == "__main__":
    cli_main()
