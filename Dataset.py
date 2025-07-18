from torch.utils.data import Dataset
from PIL import Image
import random

class ImageDataset(Dataset):
    def __init__(self,domainA_paths, domainB_paths, transform=None):
        self.domainA_paths = domainA_paths
        self.domainB_paths = domainB_paths
        self.transform = transform

        self.domainA_map = {i: path for i, path in enumerate(self.domainA_paths)}
        self.domainB_map = {i: path for i, path in enumerate(self.domainB_paths)}

        self.max_length = max(len(self.domainA_paths), len(self.domainB_paths))

    def __len__(self):
        return self.max_length * 2

    def __getitem__(self, path):
        a_idx = random.randint(0, len(self.domainA_paths) - 1)
        b_idx = random.randint(0, len(self.domainB_paths) - 1)

        imgA_path = self.domainA_map[a_idx]
        imgB_path = self.domainB_map[b_idx]

        imgA = Image.open(imgA_path).convert("RGB")
        imgB = Image.open(imgB_path).convert("RGB")

        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return imgA, imgB

