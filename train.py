import pickle
import torch
from torch import nn
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, StratifiedShuffleSplit, ShuffleSplit, GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import skorch
from tripletnet.losses import CorrelationMatrixLoss, DistanceCorrelationLoss
from siamese_triplet.losses import OnlineTripletLoss
from tripletnet.networks import TripletNetwork, TripletEnsembleNetwork
from tripletnet.datahandler import BalancedDataLoader
from torchvision import transforms
from tripletnet.networks import EmbeddingNetMNIST, lmelloEmbeddingNet
import numpy as np
import pandas as pd
from tripletnet.datahandler import BasicTorchDataset
from time import time
from tripletnet.callbacks import LoadEndState
import seaborn as sns
import matplotlib.pyplot as plt
from plotEmbeddingCallback import plotEmbeddingCallback

np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def loadRPDBCSData(data_dir='data/data_classified_v6', nsigs=100000):
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=nsigs, npoints=10800, dtype=np.float32)
    targets, targets_name = D.getMulticlassTargets()
    print(targets_name)
    D.remove(targets[targets == 3])
    D.normalize(37.28941975)
    D.shuffle()

    Feats = D.asMatrix()[:, :6100]
    targets, targets_name = D.getMulticlassTargets()
    group_ids = D.groupids('bcs')

    return np.expand_dims(Feats, axis=1), targets, group_ids


def main(save_file, D, method="orig"):
    X, Y, group_ids = D
    checkpoint_callback = skorch.callbacks.Checkpoint(dirname='mysaves/', monitor='train_loss_best')
    parameters = {
        'callbacks': [checkpoint_callback, LoadEndState(checkpoint_callback), plotEmbeddingCallback('figs')],
        'max_epochs': 100,
        'batch_size': 80,
        'margin_decay_delay': 0,
        'margin_decay_value': 0.75}
    if(method == 'orig'):
        tripletnet = TripletNetwork(lmelloEmbeddingNet,
                                    optimizer=torch.optim.Adam, optimizer__lr=1e-4, optimizer__weight_decay=1e-4,
                                    module__num_outputs=8, device='cuda',
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
    tripletnet.fit(X, Y)
    classifier = Pipeline([('encodding', tripletnet),
                           ('classifier', RandomForestClassifier(100, random_state=1))])

    # sksampler = StratifiedKFold(10, shuffle=False)
    # sksampler = GroupShuffleSplit(n_splits=1,
    #                               test_size=1/min(20, len(np.unique(group_ids))),
    #                               random_state=1)
    # sksampler = GroupKFold(min(10, len(np.unique(group_ids))))
    sksampler = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    scoring = ['accuracy', 'f1_macro']
    scores = cross_validate(classifier, X, Y, groups=group_ids, scoring=scoring,
                            cv=sksampler, return_estimator=True)
    for sc in scoring:
        print("%s:%.2f%%" % (sc, scores["test_"+sc].mean()*100))
    if(save_file is not None):
        with open(save_file+"-results.csv", 'w') as f:
            for sc in scoring:
                f.write(sc+";")
                f.write(";".join([str(s) for s in scores["test_"+sc]])+'\n')

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
    # parser.add_argument('--model', type=str, required=False, help='pre-trained model in pkl')
    parser.add_argument('-o', '--outfile', type=str, required=False)
    parser.add_argument('--method', type=str, choices=['orig', 'divconquer'], default='orig')
    args = parser.parse_args()

    D = loadRPDBCSData(args.inputdata)
    main(args.outfile, D, method=args.method)
    # main2(args.outfile, D, trained_model=args.model)

'''
accuracy:94.44%
f1_macro:84.83%
'''
