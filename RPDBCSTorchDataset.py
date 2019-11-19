import torch
from sklearn.preprocessing import StandardScaler
import numpy as np


class RPDBCSTorchDataset(torch.utils.data.Dataset):
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
        self.data = torch.unsqueeze(torch.tensor(M, dtype=torch.float32), 1)
        targets, self.targets_name = rpdbcs_data.getMulticlassTargets()
        targets = targets[begin:end]
        self.targets = torch.tensor(targets.values, dtype=torch.long)
        self.train = train

    def __getitem__(self, i):
        label = self.targets[i].item()
        sample = self.data[i]
        return sample, label

    def __len__(self):
        return len(self.data)

    def getScalerParameters(self):
        return (self.scaler.mean_, self.scaler.var_**0.5)
