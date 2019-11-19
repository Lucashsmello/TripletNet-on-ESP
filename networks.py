import torch.nn as nn
import torch
import numpy as np


# def f(n):
#     c1=(n-9)/4
#     c2=(c1-9)/4
#     c3=(c2-9)/4
#     c4=(c3-9)/4
#     return (c1,c2,c3,c4)


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
    def __init__(self):
        super(lmelloEmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 16, 5), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4),
                                     nn.Conv1d(16, 32, 5), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4),
                                     nn.Conv1d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4))

        self.fc = nn.Sequential(nn.Linear(64 * 171, 128),
                                nn.PReLU(),
                                nn.Dropout(p=0.1),
                                nn.Linear(128, 64),
                                nn.PReLU(),
                                nn.Dropout(p=0.1),
                                nn.Linear(64, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


def extract_embeddings(dataloader, model, use_cuda=True):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for samples, target in dataloader:
            if use_cuda:
                samples = samples.cuda()
            embeddings[k:k+len(samples)] = model.get_embedding(samples).data.cpu().numpy()
            labels[k:k+len(samples)] = target.numpy()
            k += len(samples)
    return embeddings, labels
########################
