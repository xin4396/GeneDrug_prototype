from typing import List, Optional, Callable, Dict, Any, Tuple
import h5py
from torch.utils.data import Dataset
import bisect
import numpy as np


class H5ADDataset(Dataset):
    """
    memory-mapped dataset loader for a H5AD file
    """

    def __init__(
        self,
        h5ad_path: str,
        component_keys: Optional[List[Tuple[str, str]]] = None,
        transform: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None,
    ):
        """
        Args:
            h5ad_path: Path to the H5AD file
            component_keys: List of (component, key) tuples to include
            transform: Transform to apply to each data point
        """
        self.h5ad_path = h5ad_path

        # Open H5AD file in read-only mode
        self.h5f = h5py.File(self.h5ad_path, mode="r")

        self.X_data = self.h5f["X"]["data"]
        self.X_indices = self.h5f["X"]["indices"]
        self.X_indptr = self.h5f["X"]["indptr"]
        self.X_shape = self.h5f["X"].attrs["shape"]
        self.n_cells = self.X_shape[0]
        self.n_genes = self.X_shape[1]

        self.component_key_index = {}

        for component, key in component_keys or []:
            self.index_component_key(component, key)

        self.transform = transform

    def index_component_key(self, component: str, key: str):
        categories = list(self.h5f[component][key]["categories"])
        if not component in self.component_key_index:
            self.component_key_index[component] = {}
        if not key in self.component_key_index[component]:
            self.component_key_index[component][key] = {}
        for category_index, category in enumerate(categories):
            self.component_key_index[component][key][category_index] = category

    def get_X_i(self, i):
        start_idx = self.X_indptr[i]
        end_idx = self.X_indptr[i + 1]

        # these two lines are the bottleneck
        row_data = self.X_data[start_idx:end_idx]
        row_indices = self.X_indices[start_idx:end_idx]

        row = np.zeros(self.n_genes)
        row[row_indices] = row_data
        return row

    def get_component_key_i(self, component: str, key: str, i: int):
        code = self.h5f[component][key]["codes"][i]
        return self.component_key_index[component][key][code]

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load main data matrix row
        X = self.get_X_i(idx)

        data_point = {
            "X": X,
        }

        for component in self.component_key_index:
            component_data = {}
            for key in self.component_key_index[component]:
                component_data[key] = self.get_component_key_i(component, key, idx)
            data_point[component] = component_data

        if self.transform:
            data_point = self.transform(data_point, idx)

        return data_point


class MultiH5ADDataset(Dataset):
    """
    memory-mapped dataset loader for a list of H5AD files
    """

    def __init__(
        self,
        h5ad_paths: List[str],
        component_keys: Optional[List[Tuple[str, str]]] = None,
        transform: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None,
    ):
        """
        Args:
            h5ad_paths: List of paths to the H5AD files
            component_keys: List of (component, key) tuples to include
            transform: Transform to apply to each data point
        """
        self.datasets: List[H5ADDataset] = []
        for h5ad_path in h5ad_paths:
            self.datasets.append(
                H5ADDataset(
                    h5ad_path,
                    component_keys=component_keys,
                    transform=transform,
                )
            )

        ## create cumulative lengths for indexing
        self.cumulative_lengths = []
        cum_len = 0
        for dataset in self.datasets:
            cum_len += len(dataset)
            self.cumulative_lengths.append(cum_len)

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._get_single_item(idx)

        # if isinstance(idx, int):
        #     return self._get_single_item(idx)
        # elif isinstance(idx, list) and all(isinstance(i, int) for i in idx):
        #     return [self._get_single_item(i) for i in idx]
        # else:
        #     raise TypeError("Key must be an integer or a list of integers")

    def _get_single_item(self, idx: int) -> Dict[str, Any]:
        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_lengths[dataset_idx - 1]

        return self.datasets[dataset_idx].__getitem__(sample_idx)
