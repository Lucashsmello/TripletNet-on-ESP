import torch.nn as nn
import torch
import numpy as np
from siamese_triplet.losses import OnlineTripletLoss
from siamese_triplet.trainer import train_epoch
from .trainer import train_tripletNetworkAdvanced
import siamese_triplet.trainer
from .datahandler import BasicTorchDataset
from siamese_triplet.utils import RandomNegativeTripletSelector, HardestNegativeTripletSelector, SemihardNegativeTripletSelector, HardNegativeTripletSelector
from sklearn.base import BaseEstimator, TransformerMixin


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


class TripletNetwork(BaseEstimator, TransformerMixin):
    def __init__(self, net_arch,
                 learning_rate=1e-3, num_subepochs=10, num_epochs=10, batch_size=32, dont_train=False,
                 custom_trainepoch=train_epoch,
                 custom_loss=OnlineTripletLoss,
                 triplet_selector=RandomNegativeTripletSelector):
        self.net_arch = net_arch
        self.learning_rate = learning_rate
        self.num_subepochs = num_subepochs
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.custom_loss = custom_loss
        self.custom_trainepoch = custom_trainepoch
        self.dont_train = dont_train
        self.triplet_selector = triplet_selector

    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "num_subepochs": self.num_subepochs,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "custom_trainepoch": self.custom_trainepoch,
            "custom_loss": self.custom_loss,
            "net_arch": self.net_arch,
            "dont_train": self.dont_train,
            "triplet_selector": self.triplet_selector
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if(parameter == 'learning_rate'):
                self.learning_rate = value
            elif(parameter == 'num_subepochs'):
                self.num_subepochs = value
            elif(parameter == 'batch_size'):
                self.batch_size = value
            elif(parameter == 'num_epochs'):
                self.num_epochs = value
            elif(parameter == 'net_arch'):
                self.net_arch = value
            elif(parameter == 'custom_loss'):
                self.custom_loss = value
            elif(parameter == 'custom_trainepoch'):
                self.custom_trainepoch = value
            elif(parameter == 'dont_train'):
                self.dont_train = value
            print("Parameter %s not recognized by TripletNetwork!" % parameter)
        return self

    def fit(self, X, y=None):
        if(not self.dont_train):
            if(isinstance(X, torch.utils.data.Dataset)):
                D = X
            else:
                D = (X, y)
            self.train(D, self.learning_rate, self.num_subepochs, self.batch_size, self.num_epochs,
                       custom_loss=self.custom_loss,
                       custom_trainepoch=self.custom_trainepoch)
        return self

    def transform(self, X):
        return self.embed(X).cpu().numpy()

    def train(self, D, learning_rate, num_subepochs, batch_size=16, num_epochs=16,
              custom_loss=OnlineTripletLoss,
              custom_trainepoch=siamese_triplet.trainer.train_epoch,
              triplet_selector=RandomNegativeTripletSelector):
        margin1 = 1.0
        triplet_train_config = [
            {'triplet-selector': triplet_selector,
             'learning-rate': learning_rate,
             'margin': margin1,
             'nepochs': num_subepochs
             }
        ]
        train_tripletNetworkAdvanced(
            D, None, self.net_arch, triplet_train_config,
            gamma=0.1, beta=0.25, niterations=num_epochs, batch_size=batch_size,
            loss_function_generator=custom_loss, custom_trainepoch=custom_trainepoch)

    def embed(self, X):
        """
        Transform features from the original to the triplet-space.

        Args:
            X: Each line contains an object. Should be a tensor or a torch.utils.data.Dataset that returns tensors.

        Returns:
            The encodding in a pytorch.tensor matrix of len(X) lines.
        """
        if(not isinstance(X, torch.utils.data.Dataset)):
            D = BasicTorchDataset(X, None, single_channel=True)
        else:
            D = X
        kwargs = {'num_workers': 3, 'pin_memory': True}
        dataloader = torch.utils.data.DataLoader(D, batch_size=32, **kwargs)

        with torch.no_grad():
            if(hasattr(self.net_arch, 'num_outputs')):
                num_outputs = self.net_arch.num_outputs
            else:
                for last_module in self.net_arch.modules():
                    pass
                num_outputs = last_module.out_features
            ret = torch.empty((len(X), num_outputs), dtype=torch.float32).cuda()
            k = 0
            for x in dataloader:
                if(isinstance(x, tuple) or isinstance(x, list)):
                    x = x[0]
                x = x.cuda()
                output = self.net_arch.forward(x)
                ret[k:k+len(output)] = output
                k += len(output)
        return ret

    def save(self, fpath):
        data_to_save = {'state_dict': self.net_arch.state_dict(),
                        'net_arch_name': self.net_arch.__class__.__name__}
        torch.save(data_to_save, fpath)

    @staticmethod
    def load(fpath, net_arch, map_location=None) -> 'TripletNetwork':
        checkpoint = torch.load(fpath, map_location=map_location)
        if('net_arch_name' in checkpoint):
            msg = "Network arch in %s is incompatible with %s."
            msg = msg % (fpath, net_arch.__class__.__name__)
            assert(checkpoint['net_arch_name'] == net_arch.__class__.__name__), msg
        net_arch.load_state_dict(checkpoint['state_dict'])
        return TripletNetwork(net_arch)

    def cuda(self):
        self.net_arch.cuda()
        return self

    def eval(self):
        self.net_arch.eval()
        return self


class lmelloEmbeddingNet(nn.Module):
    def __init__(self, num_outputs, num_inputs_channels=1):
        super().__init__()
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
        output = output.view(output.size()[0], -1)  # flatten
        output = self.fc(output)
        return output

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
