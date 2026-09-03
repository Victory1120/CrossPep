import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from utils import integer_label_protein


class DTIDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, df):
        self.list_IDs = list_IDs
        self.df = df

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d_fpcp = self.df.iloc[index]['fcfp']  # 假设是 NumPy 数组或其他可转换为张量的类型
        # v_p = self.df.iloc[index]['Protein']  # 假设是字符串，需要进一步处理
        # v_p = integer_label_protein(v_p)  # 将蛋白质序列转换为整数标签或其他格式
        esm = self.df.iloc[index]['esm']
        y = self.df.iloc[index]["Y"]  # 标签
        return v_d_fpcp, esm, y  # esm,

class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches

