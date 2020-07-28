import torch
from torch import nn
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from tripletnet.losses import CorrelationMatrixLoss, DistanceCorrelationLoss
from siamese_triplet.losses import OnlineTripletLoss
from tripletnet.networks import TripletNetwork
from tripletnet.datahandler import BalancedDataLoader
from torchvision import transforms
from tripletnet.networks import EmbeddingNetMNIST, lmelloEmbeddingNet2, lmelloEmbeddingNet
import numpy as np
import pandas as pd
from tripletnet.datahandler import BasicTorchDataset
from time import time


def loadRPDBCSData(nsigs=100000):
    data_dir = 'data/data_classified_v6'
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=nsigs, npoints=10800, dtype=np.float32)
    targets, targets_name = D.getMulticlassTargets()
    # print(targets_name)
    # D.remove(((targets[targets >= 2]).index).values)
    D.normalize(37.28941975)
    D.shuffle()

    Feats = D.asMatrix()[:, :6100]
    targets, targets_name = D.getMulticlassTargets()

    return np.expand_dims(Feats, axis=1), targets


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


def main(save_file, D):
    X, Y = D
    tripletnet = TripletNetwork(lmelloEmbeddingNet, margin_decay_delay=10,
                                optimizer=torch.optim.Adam, optimizer__lr=1e-3, optimizer__weight_decay=1e-4,
                                module__num_outputs=8, device='cuda',
                                train_split=None,
                                batch_size=125, max_epochs=30,
                                criterion=TripletNetwork.OnlineTripletLossWrapper,
                                iterator_train=BalancedDataLoader, iterator_train__num_workers=3, iterator_train__pin_memory=True)

    classifier = Pipeline([('encodding', tripletnet),
                           ('classifier', RandomForestClassifier(100))])

    # classifier.fit(X,Y)
    sksampler = StratifiedKFold(5, shuffle=True)
    scores = cross_validate(classifier, X, Y, scoring=['accuracy', 'f1_macro'], cv=sksampler)
    print(scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, required=True)
    args = parser.parse_args()

    D = loadRPDBCSData()
    # D = loadMNIST()
    main(args.outfile, D)
