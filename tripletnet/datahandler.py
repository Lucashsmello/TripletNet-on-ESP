from typing import Iterable
import torch
from siamese_triplet.datasets import BalancedBatchSampler
import numpy as np


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

class MyBalancedBatchSampler(BalancedBatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        if(torch.is_tensor(labels)):
            labels = labels.numpy()
        self.labels_set = list(set(labels))
        self.label_to_indices = {label: np.where(labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(labels)
        self.batch_size = self.n_samples * self.n_classes


class BalancedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, num_workers=0, collate_fn=None,
                 pin_memory=False, worker_init_fn=None):
        nclasses = len(set(dataset.y))
        sampler = MyBalancedBatchSampler(dataset.y, nclasses, batch_size//nclasses)
        super().__init__(dataset, num_workers=num_workers, batch_sampler=sampler,
                         collate_fn=collate_fn, pin_memory=pin_memory, worker_init_fn=worker_init_fn)