import torch
from sklearn.preprocessing import StandardScaler


class RPDBCSTorchDataset(torch.utils.data.Dataset):
    def __init__(self, rpdbcs_data, train, signal_size):
        M = rpdbcs_data.asMatrix()[:, :signal_size]
        self.scaler = StandardScaler()
        self.scaler.fit(M)
        M = self.scaler.transform(M)
        self.data = torch.unsqueeze(torch.tensor(M, dtype=torch.float32), 1)
        targets, self.targets_name = rpdbcs_data.getMulticlassTargets()
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
