import torch
from sklearn.preprocessing import StandardScaler
import numpy as np


def fitScaler(M):
    scaler = StandardScaler()
    scaler.fit(M)
    # print(self.getScalerParameters())
    allmean = np.mean(scaler.mean_)
    allvar = np.mean([v**0.5 for v in scaler.var_])
    allvar = allvar**2
    for i in range(len(scaler.mean_)):
        scaler.mean_[i] = 0
        scaler.var_[i] = 1

    return scaler


class BasicTorchDataset(torch.utils.data.Dataset):
    def __init__(self, feats_matrix, targets, targets_name=None):
        self.data = torch.unsqueeze(torch.tensor(feats_matrix, dtype=torch.float32), 1)
        self.targets_orig = targets
        if(targets is not None):
            self.targets = torch.tensor(targets, dtype=torch.long)
        else:
            self.targets = None
        self.targets_name = targets_name
        self.bootstrap_permutation = None
        if(targets is not None):
            self.nclasses = len(set(targets))

    def getTargets(self):
        if(self.bootstrap_permutation is None):
            return self.targets
        return torch.tensor(self.targets_orig[self.bootstrap_permutation], dtype=torch.long)

    def __getitem__(self, i):
        if(self.bootstrap_permutation is not None):
            i = self.bootstrap_permutation[i]
        if(self.targets is None):
            label = -1
        else:
            label = self.targets[i].item()
        sample = self.data[i]
        return sample, label

    def __len__(self):
        if(self.bootstrap_permutation is None):
            return len(self.data)
        return len(self.bootstrap_permutation)

    def setBootstrap(self, b):
        n = len(self.data)
        self.bootstrap_permutation = np.random.permutation(n)[:int(n*(1-b))]


class RPDBCSTorchDataset(BasicTorchDataset):
    def __init__(self, rpdbcs_data, train, signal_size, scaler=None, holdout=0.67):
        holdout_index = int(holdout*len(rpdbcs_data))
        if(train):
            begin = 0
            end = holdout_index
        else:
            begin = holdout_index
            end = len(rpdbcs_data)
        M = rpdbcs_data.asMatrix()[begin:end, :signal_size]
        if(scaler is None):
            self.scaler = fitScaler(M)
        else:
            self.scaler = scaler
        M = self.scaler.transform(M)
        targets, targets_name = rpdbcs_data.getMulticlassTargets()
        targets = targets[begin:end]
        super().__init__(M, targets.values, targets_name)
        self.train = train

    def getScalerParameters(self):
        return (self.scaler.mean_, self.scaler.var_**0.5)
