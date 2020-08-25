import pickle
import torch
from torch import nn
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import skorch
from tripletnet.losses import CorrelationMatrixLoss, DistanceCorrelationLoss
from siamese_triplet.losses import OnlineTripletLoss
from tripletnet.networks import TripletNetwork, TripletEnsembleNetwork
from tripletnet.datahandler import BalancedDataLoader
from torchvision import transforms
from tripletnet.networks import EmbeddingNetMNIST, lmelloEmbeddingNet2, lmelloEmbeddingNet
import numpy as np
import pandas as pd
from tripletnet.datahandler import BasicTorchDataset
from time import time
from tripletnet.callbacks import LoadEndState

np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def loadRPDBCSData(data_dir='data/data_classified_v6', nsigs=100000):
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=nsigs, npoints=10800, dtype=np.float32)
    targets, targets_name = D.getMulticlassTargets()
    # D.remove(((targets[targets >= 2]).index).values)
    D.normalize(37.28941975)
    D.shuffle()

    Feats = D.asMatrix()[:, :6100]
    targets, targets_name = D.getMulticlassTargets()

    return np.expand_dims(Feats, axis=1), targets


def loadMNIST():
    from torchvision.datasets import MNIST

    mean, std = 0.1307, 0.3081
    dataset_train = MNIST('/tmp', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((mean,), (std,))
                          ]))
    dataset_test = MNIST('/tmp', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((mean,), (std,))
                         ]))

    return dataset_train, dataset_test


def main(save_file, D, method="orig"):
    X, Y = D
    checkpoint_callback = skorch.callbacks.Checkpoint(dirname='mysaves/', monitor='train_loss_best')
    parameters = {'callbacks': [checkpoint_callback, LoadEndState(checkpoint_callback)],
                  'max_epochs': 20*16,
                  'batch_size': 125,
                  'margin_decay_delay': 20,
                  'margin_decay_value': 0.75}
    if(method == 'orig'):
        tripletnet = TripletNetwork(lmelloEmbeddingNet,
                                    optimizer=torch.optim.Adam, optimizer__lr=1e-4, optimizer__weight_decay=1e-4,
                                    module__num_outputs=16, device='cuda',
                                    train_split=None,
                                    criterion=TripletNetwork.OnlineTripletLossWrapper,
                                    iterator_train=BalancedDataLoader, iterator_train__num_workers=3, iterator_train__pin_memory=True,
                                    **parameters
                                    )
    else:
        tripletnet = TripletEnsembleNetwork(lmelloEmbeddingNet, k=4,
                                            optimizer=torch.optim.Adam, optimizer__lr=1e-4, optimizer__weight_decay=1e-4,
                                            module__num_outputs=32, device='cuda',
                                            train_split=None,
                                            criterion=TripletNetwork.OnlineTripletLossWrapper,
                                            iterator_train=BalancedDataLoader, iterator_train__num_workers=3, iterator_train__pin_memory=True,
                                            **parameters)

    classifier = Pipeline([('encodding', tripletnet),
                           ('classifier', RandomForestClassifier(100))])

    # classifier.fit(X,Y)
    sksampler = StratifiedKFold(10, shuffle=False)
    # sksampler = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    scoring = ['accuracy', 'f1_macro']
    scores = cross_validate(classifier, X, Y, scoring=scoring, cv=sksampler, return_estimator=True)
    with open(save_file+"-results.csv", 'w') as f:
        for sc in scoring:
            f.write(sc+";")
            f.write(";".join([str(s) for s in scores["test_"+sc]])+'\n')
            print("%s:%.2f%%" % (sc, scores["test_"+sc].mean()*100))
    for i, trained_model in enumerate(scores['estimator']):
        trained_model['encodding'].save_params("%s-%d.pt" % (save_file, i))
        # trained_model = trained_model['encodding']
        # with open("%s-%d.pkl" % (save_file, i), 'wb') as f:
        #     pickle.dump(trained_model, f)


def main2(save_file, D, trained_model=None):
    X, Y = D
    n = 4500
    if(trained_model is None):
        checkpoint_callback = skorch.callbacks.Checkpoint(dirname='mysaves/')
        tripletnet = TripletNetwork(lmelloEmbeddingNet, margin_decay_delay=0,
                                    optimizer=torch.optim.Adam, optimizer__lr=1e-4, optimizer__weight_decay=1e-4,
                                    module__num_outputs=8, device='cuda',
                                    # train_split=None,
                                    batch_size=125, max_epochs=5,
                                    criterion=TripletNetwork.OnlineTripletLossWrapper,
                                    iterator_train=BalancedDataLoader, iterator_train__num_workers=3, iterator_train__pin_memory=True,
                                    callbacks=[checkpoint_callback, LoadEndState(checkpoint_callback)])

    else:
        tripletnet = TripletNetwork.load(trained_model, module=lmelloEmbeddingNet,
                                         module__num_outputs=8, device='cpu')
        tripletnet.dont_train = True

    classifier = Pipeline([('encodding', tripletnet),
                           ('classifier', KNeighborsClassifier(1))])
    classifier.fit(X[:n], Y[:n])
    preds = classifier.predict(X[n:])
    print("Accuracy:", accuracy_score(Y[n:], preds))
    print("fmeasure:", f1_score(Y[n:], preds, average='macro'))

    if(save_file is not None):
        tripletnet.save_params(save_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdata', type=str, required=True)
    parser.add_argument('--model', type=str, required=False, help='pre-trained model in pkl')
    parser.add_argument('-o', '--outfile', type=str, required=False)
    parser.add_argument('--method', type=str, choices=['orig', 'divconquer'], default='orig')
    args = parser.parse_args()

    D = loadRPDBCSData(args.inputdata)
    main(args.outfile, D, method=args.method)
    # main2(args.outfile, D, trained_model=args.model)
