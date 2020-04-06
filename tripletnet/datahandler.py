from typing import Iterable
import torch


class BasicTorchDataset(torch.utils.data.Dataset):
    def __init__(self, feats_matrix, targets: Iterable[int]):
        self.data = torch.unsqueeze(torch.tensor(feats_matrix, dtype=torch.float32), 1)
        self.targets_orig = targets
        if(targets is not None):
            self.targets = torch.tensor(targets, dtype=torch.long)
        else:
            self.targets = None
        if(targets is not None):
            self.nclasses = len(set(targets))

    def getTargets(self):
        return self.targets

    def __getitem__(self, i):
        if(self.targets is None):
            label = -1
        else:
            label = self.targets[i].item()
        sample = self.data[i]
        return sample, label

    def __len__(self):
        return len(self.data)


class RPDBCSTorchDataset(BasicTorchDataset):
    def __init__(self, rpdbcs_data, signal_size: int, train=True, holdout: float = 1.0):
        holdout_index = int(holdout*len(rpdbcs_data))
        if(train):
            begin = 0
            end = holdout_index
        else:
            begin = holdout_index
            end = len(rpdbcs_data)
        M = rpdbcs_data.asMatrix()[begin:end, :signal_size]
        targets, targets_name = rpdbcs_data.getMulticlassTargets()
        targets = targets[begin:end]
        super().__init__(M, targets.values, targets_name)
        self.train = train

    def getScalerParameters(self):
        return (self.scaler.mean_, self.scaler.var_**0.5)
