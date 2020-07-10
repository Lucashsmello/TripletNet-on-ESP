from typing import Iterable
import torch


class BasicTorchDataset(torch.utils.data.TensorDataset):
    def __init__(self, features, targets, single_channel=False):
        tensors = torch.tensor(features, dtype=torch.float32)
        if(single_channel):
            tensors = tensors.unsqueeze(dim=1)
        if(targets is not None):
            self.targets = torch.tensor(targets)
            super().__init__(tensors, self.targets)
        else:
            super().__init__(tensors)
