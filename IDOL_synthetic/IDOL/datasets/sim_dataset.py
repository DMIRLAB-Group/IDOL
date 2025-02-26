import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import ipdb as pdb

class StationaryDataset(Dataset):
    
    def __init__(self, dataset):
        super().__init__()
        self.path = os.path.join('../IDOL/datasets/dataset', dataset, "data.npz")
        self.npz = np.load(self.path)
        self.data = { }
        for key in ["yt", "xt"]:
            self.data[key] = self.npz[key]

    def __len__(self):
        return len(self.data["yt"])

    def __getitem__(self, idx):
        yt = torch.from_numpy(self.data["yt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        sample = {"yt": yt, "xt": xt}
        return sample
