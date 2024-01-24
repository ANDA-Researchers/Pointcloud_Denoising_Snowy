from pytorch_lightning.cli import LightningCLI

from pc_denoising.data_module import KITTI
from pc_denoising.model import MinkowskiUNet


def cli_main():
    cli = LightningCLI(MinkowskiUNet, KITTI)


if __name__ == "__main__":
    cli_main()
