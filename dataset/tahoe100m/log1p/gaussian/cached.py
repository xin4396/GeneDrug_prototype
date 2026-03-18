import pickle
from typing import Callable, Dict, Optional
from pyparsing import Any
from torch.utils.data import Dataset
import numpy as np


class CachedGaussianDataset(Dataset):
    def __init__(
        self,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        with open("/home/mark/repos/genedrug/cache/dataset/tahoe100m/log1p/gaussian/metadata.pkl", "rb") as f:
            metas = pickle.load(f)

        with np.load("/home/mark/repos/genedrug/cache/dataset/tahoe100m/log1p/gaussian/mean_std.npz") as data:
            log1p_mean = data["mean"]
            log1p_std = data["std"]
            mask = data["mask"]

        self.metas = []
        self.results = []

        for i in range(len(metas)):
            if mask[i] == True:
                self.metas.append(metas[i])
                self.results.append((log1p_mean[i], log1p_std[i]))

        self.transform = transform

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        result = self.results[idx]
        plate, cell_line_str, drug_name, drug_conc = meta
        data_point = {
            "plate": plate,
            "cell_line": cell_line_str,
            "drug_name": drug_name,
            "drug_conc": drug_conc,
            "log1p_mean": result[0],
            "log1p_std": result[1],
        }

        if self.transform:
            data_point = self.transform(data_point, idx)

        return data_point
