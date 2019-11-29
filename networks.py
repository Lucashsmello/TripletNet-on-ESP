import torch.nn as nn
import torch
import numpy as np


class LarsHuyEmbeddingNet(nn.Module):
    def __init__(self):
        super(LarsHuyEmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 48, 9), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4),
                                     nn.Conv1d(48, 96, 9), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4),
                                     nn.Conv1d(96, 192, 9), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4),
                                     nn.Conv1d(192, 384, 9), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4)
                                     )

        self.fc = nn.Sequential(nn.Linear(384*40, 384),
                                nn.PReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(384, 384),
                                nn.PReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(384, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class lmelloEmbeddingNet(nn.Module):
    def __init__(self, num_outputs=2):
        super(lmelloEmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 16, 5), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4),
                                     nn.Conv1d(16, 32, 5), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4),
                                     nn.Conv1d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4)
                                     )

        self.fc = nn.Sequential(nn.Linear(64 * 171, 128),
                                nn.PReLU(),
                                nn.Dropout(p=0.1),
                                nn.Linear(128, 64),
                                nn.PReLU(),
                                nn.Dropout(p=0.1),
                                nn.Linear(64, num_outputs)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


def extract_embeddings(dataloader, model, num_outputs=-1, use_cuda=True, with_labels=True):
    if(num_outputs <= 0):
        for last_module in model.modules():
            pass
        num_outputs = last_module.out_features
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), num_outputs))
        if(with_labels):
            labels = np.zeros(len(dataloader.dataset))
        k = 0
        for samples, target in dataloader:
            if use_cuda:
                samples = samples.cuda()
            embeddings[k:k+len(samples)] = model.get_embedding(samples).data.cpu().numpy()
            if(with_labels):
                labels[k:k+len(samples)] = target.numpy()
            k += len(samples)
    if(with_labels):
        return embeddings, labels
    return embeddings
########################
