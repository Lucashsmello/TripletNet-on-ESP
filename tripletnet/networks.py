import torch.nn as nn
import torch
import numpy as np


def initWeights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class EmbeddingNetMNIST(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        # input: (1,28,28)
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.2),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )

        self.fc = nn.Sequential(nn.Linear(32*14*14, 512), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(512, num_outputs)
                                )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


class BrunaEmbeddingNet(nn.Module):
    def __init__(self, num_outputs=2):
        super(BrunaEmbeddingNet, self).__init__()
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
        self.fc = nn.Sequential(nn.Linear(64 * 378, num_outputs))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class lmelloEmbeddingNet(nn.Module):
    def __init__(self, num_outputs, num_inputs_channels=1):
        super(lmelloEmbeddingNet, self).__init__()
        self.num_outputs = num_outputs
        self.convnet = nn.Sequential(
            nn.Conv1d(num_inputs_channels, 16, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(32, 64, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4)
        )

        self.fc = nn.Sequential(nn.Linear(64 * 94, 192),
                                nn.LeakyReLU(negative_slope=0.05),
                                nn.Linear(192, num_outputs)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def encode(self, x):
        with torch.no_grad():
            if(len(x.shape) == 1):
                n = 1
                x = torch.tensor(x[:6100], dtype=torch.float32).cuda()
                x = x.reshape((1, 1, 6100))
                return self.forward(x).squeeze()
            else:
                n = x.shape[0]
                ret = torch.empty((n, self.num_outputs), dtype=torch.float32).cuda()
                k = 0
                for i in range(0, n, 8):
                    batch = torch.tensor(x[i:i+8, :6100], dtype=torch.float32).cuda()
                    batch = batch.reshape(batch.shape[0], 1, 6100)
                    output = self.forward(batch).squeeze()
                    ret[k:k+len(output)] = output
                    k += len(output)
                return ret

            # if(next(self.parameters()).is_cuda):
            #     x = x.cuda()


class lmelloEmbeddingNet2(lmelloEmbeddingNet):
    def __init__(self, num_outputs, num_inputs_channels=1):
        super().__init__(num_outputs, num_inputs_channels)
        self.convnet = nn.Sequential(
            nn.Conv1d(num_inputs_channels, 16, 5), nn.ReLU(), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, 5), nn.ReLU(), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(32, 64, 5), nn.ReLU(), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4)
        )

        self.fc = nn.Sequential(nn.Linear(64 * 94, 192),
                                nn.ReLU(),
                                nn.Linear(192, num_outputs)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class lmelloOnlyConvNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        # input size: 6100
        self.convnet = nn.Sequential(
            nn.Conv1d(1, 16, 5), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, 5), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(32, 64, 5), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),  # out: 94
            nn.Conv1d(64, 128, 5), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(128, 256, 3, padding=1), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.Conv1d(256, 128, 3, padding=1), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.2),
            nn.Conv1d(128, num_outputs, 45)  # out: 45
        )

    def forward(self, x):
        output = self.convnet(x)
        return output.reshape(x.shape[0], output.shape[1])


class lmelloEmbeddingNetReducedFC(lmelloEmbeddingNet):
    def __init__(self, num_outputs):
        super().__init__(num_outputs)
        self.convnet = nn.Sequential(
            nn.Conv1d(1, 16, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(32, 64, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4)
        )

        self.fc = nn.Sequential(nn.Linear(64 * 94, num_outputs),
                                )


class lmelloEmbeddingNetReducedConv(lmelloEmbeddingNet):
    def __init__(self, num_outputs):
        super().__init__(num_outputs)
        self.convnet = nn.Sequential(
            nn.Conv1d(1, 16, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(16, 32, 5), nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(8, stride=8)
        )

        self.fc = nn.Sequential(nn.Linear(32 * 190, 192),
                                nn.LeakyReLU(negative_slope=0.05),
                                nn.Linear(192, num_outputs)
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
            embeddings[k:k+len(samples)] = model.forward(samples).data.cpu().numpy()
            if(with_labels):
                labels[k:k+len(samples)] = target.numpy()
            k += len(samples)
    if(with_labels):
        return embeddings, labels
    return embeddings
########################

# from torchsummary import summary

# net=lmelloEmbeddingNet(8)
# net.cuda()
# summary(net, input_size=(1, 6100))
