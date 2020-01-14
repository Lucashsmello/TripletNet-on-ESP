import torch.nn as nn
import torch
import numpy as np


def initWeights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class BrunaEmbeddingNet(nn.Module):
    def __init__(self, num_outputs=2, num_knownfeats=0):
        super(BrunaEmbeddingNet, self).__init__()
        self.num_knownfeats = num_knownfeats
        self.convnet = nn.Sequential(
            nn.Conv1d(1, 16, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Conv1d(16, 16, 1), nn.LeakyReLU(negative_slope=0.05),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, 3), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.Conv1d(32, 32, 3), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(32, 64, 3), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.Conv1d(64, 64, 1), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2)
        )
        self.fc = nn.Sequential(nn.Linear(64 * 378+num_knownfeats, num_outputs))

    def forward(self, x):
        if(self.num_knownfeats > 0):
            oracle_feats = x[:, :, :self.num_knownfeats].squeeze(dim=1)
            output = self.convnet(x[:, :, self.num_knownfeats:])
            output = output.view(output.size()[0], -1)
            output = torch.cat([oracle_feats, output], dim=1)
            output = self.fc(output)
        else:
            output = self.convnet(x)
            output = output.view(output.size()[0], -1)
            output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class lmelloEmbeddingNet(nn.Module):
    def __init__(self, num_outputs, num_knownfeats=0):
        super(lmelloEmbeddingNet, self).__init__()
        self.num_knownfeats = num_knownfeats
        self.convnet = nn.Sequential(
            nn.Conv1d(1, 16, 5), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, 5), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(32, 64, 5), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4)
        )

        self.fc = nn.Sequential(nn.Linear(64 * 94+num_knownfeats, 192),
                                nn.LeakyReLU(negative_slope=0.05),
                                # nn.Linear(128, 64),
                                # nn.PReLU(),
                                # nn.Dropout(p=0.1),
                                nn.Linear(192, num_outputs)
                                )

    def forward(self, x):
        if(self.num_knownfeats > 0):
            oracle_feats = x[:, :, :self.num_knownfeats].squeeze(dim=1)
            output = self.convnet(x[:, :, self.num_knownfeats:])
            output = output.view(output.size()[0], -1)
            output = torch.cat([oracle_feats, output], dim=1)
            output = self.fc(output)
        else:
            output = self.convnet(x)
            output = output.view(output.size()[0], -1)
            output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class lmelloEmbeddingNet2(lmelloEmbeddingNet):
    def __init__(self, num_outputs=2, num_knownfeats=0):
        super(lmelloEmbeddingNet, self).__init__()
        self.num_knownfeats = num_knownfeats
        self.convnet = nn.Sequential(
            nn.Conv1d(1, 16, 5), nn.PReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, 5), nn.PReLU(),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(32, 64, 5), nn.PReLU(),
            nn.MaxPool1d(4, stride=4)
        )

        self.fc = nn.Sequential(nn.Linear(64 * 94+num_knownfeats, 192),
                                nn.PReLU(),
                                nn.Linear(192, 64),
                                nn.PReLU(),
                                nn.Linear(64, num_outputs)
                                )


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
