import torch
from sklearn.preprocessing import StandardScaler
import numpy as np


def fitScaler(M):
    scaler = StandardScaler()
    scaler.fit(M)
    # print(self.getScalerParameters())
    allmean = np.mean(scaler.mean_)
    allvar = np.mean(scaler.var_)
    for i in range(len(scaler.mean_)):
        scaler.mean_[i] = allmean
        scaler.var_[i] = allvar

    return scaler


class BasicTorchDataset(torch.utils.data.Dataset):
    def __init__(self, feats_matrix, targets, targets_name=None):
        self.data = torch.unsqueeze(torch.tensor(feats_matrix, dtype=torch.float32), 1)
        if(targets is not None):
            self.targets = torch.tensor(targets, dtype=torch.long)
        else:
            self.targets = None
        self.targets_name = targets_name

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
