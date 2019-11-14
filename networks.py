import torch.nn as nn


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
        self.convnet = nn.Sequential(nn.Conv1d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4),
                                     nn.Conv1d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4),
                                     nn.Conv1d(64, 128, 5), nn.PReLU(),
                                     nn.MaxPool1d(4, stride=4))

        self.fc = nn.Sequential(nn.Linear(128 * 171, 128),
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
