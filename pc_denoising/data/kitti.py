import pickle
from pathlib import Path

from torch.utils.data import Dataset


class KITTI(Dataset):
    def __init__(self, lidar_dir: Path, info_path: Path) -> None:
        super().__init__()
        infos = pickle.load(info_path.read_bytes())
        print(infos)
