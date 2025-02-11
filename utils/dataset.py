import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


# Latent dataset loader for .npy files
class LatentDataset(Dataset):
    def __init__(self, latent_dir, transform=None):
        self.latent_dir = Path(latent_dir)
        self.latent_files = sorted(self.latent_dir.glob("*.npy"))
        if not self.latent_files:
            raise ValueError(f"No .npy files found in: {latent_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_file = self.latent_files[idx]
        latent = np.load(latent_file)
        latent_tensor = torch.from_numpy(latent).float()
        if self.transform:
            latent_tensor = self.transform(latent_tensor)
        return latent_tensor