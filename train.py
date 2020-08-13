import torch
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset
from tripletnet.classifiers.augmented_classifier import ClassifierConvNet, AugmentedClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tripletnet.losses import CorrelationMatrixLoss, DistanceCorrelationLoss
from siamese_triplet.losses import OnlineTripletLoss
from siamese_triplet.utils import HardestNegativeTripletSelector
from tripletnet.networks import TripletNetwork
from torchvision import transforms
from tripletnet.networks import EmbeddingNetMNIST, lmelloEmbeddingNet2, lmelloEmbeddingNet
import numpy as np
import pandas as pd
from tripletnet.datahandler import BasicTorchDataset
from torch.utils.data.sampler import BatchSampler
from sklearn.model_selection import cross_validate

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def loadRPDBCSData(data_dir, nsigs=100000):
    # data_dir = 'data/data_classified_v6'
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=nsigs, npoints=10800)
    targets, targets_name = D.getMulticlassTargets()
    D.normalize(37.28941975)
    # D.shuffle()

    Feats = D.asMatrix()[:, :6100]
    targets, targets_name = D.getMulticlassTargets()

    return Feats, targets


def loadMNIST():
    from torchvision.datasets import MNIST

    mean, std = 0.1307, 0.3081
    dataset = MNIST('/tmp', train=train, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((mean,), (std,))
                    ]))
    dataset_test = MNIST('/tmp', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((mean,), (std,))
                         ]))

    return dataset, dataset_test


def main(D):
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import Pipeline

    metrics = {'Accuracy': accuracy_score,
               'f-macro': lambda x, y: f1_score(x, y, average='macro'),
               'prec-macro': lambda x, y: precision_score(x, y, average='macro'),
               'recall-macro': lambda x, y: recall_score(x, y, average='macro')}
    X, Y = D

    net = TripletNetwork(net_arch=lmelloEmbeddingNet(8).cuda(),
                         learning_rate=1e-3, num_subepochs=3, batch_size=25, num_epochs=3,
                         custom_loss=OnlineTripletLoss,
                         triplet_selector=HardestNegativeTripletSelector
                         )
    classifier = Pipeline([('embedder', net),
                           ('classifier', KNeighborsClassifier(1))])

    classifier.fit(X, Y)
    Ypred = classifier.predict(X)
    for mname, m in metrics.items():
        score = m(Y, Ypred)
        print("%s: %f%%" % (mname, score*100))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help="input directory of dataset")
    args = parser.parse_args()

    D = loadRPDBCSData(args.input)
    main(D)
