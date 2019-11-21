import torch
from sklearn.preprocessing import StandardScaler
import numpy as np


class BasicTorchDataset(torch.utils.data.Dataset):
    def __init__(self, feats_matrix, targets, targets_name):
        self.M = feats_matrix
        self.data = torch.unsqueeze(torch.tensor(self.M, dtype=torch.float32), 1)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.targets_name = targets_name

    def __getitem__(self, i):
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
            self.scaler = StandardScaler()
            self.scaler.fit(M)
            # print(self.getScalerParameters())
            allmean = np.mean(self.scaler.mean_)
            allvar = np.mean(self.scaler.var_)
            for i in range(len(self.scaler.mean_)):
                self.scaler.mean_[i] = allmean
                self.scaler.var_[i] = allvar
        else:
            self.scaler = scaler
        M = self.scaler.transform(M)
        targets, targets_name = rpdbcs_data.getMulticlassTargets()
        targets = targets[begin:end]
        super().__init__(M, targets.values, targets_name)
        self.train = train

    def __getitem__(self, i):
        label = self.targets[i].item()
        sample = self.data[i]
        return sample, label

    def __len__(self):
        return len(self.data)

    def getScalerParameters(self):
        return (self.scaler.mean_, self.scaler.var_**0.5)
