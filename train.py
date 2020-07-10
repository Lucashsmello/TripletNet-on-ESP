import torch
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset
from tripletnet.classifiers.augmented_classifier import ClassifierConvNet, AugmentedClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tripletnet.losses import CorrelationMatrixLoss, DistanceCorrelationLoss
from siamese_triplet.losses import OnlineTripletLoss
from tripletnet.networks import TripletNetwork
from torchvision import transforms
from tripletnet.networks import EmbeddingNetMNIST, lmelloEmbeddingNet2, lmelloEmbeddingNet
import numpy as np

import pandas as pd


def loadRPDBCSData(nsigs=100000):
    data_dir = 'data/data_classified_v6'
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=nsigs, npoints=10800)
    targets, targets_name = D.getMulticlassTargets()
    # print(targets_name)
    # D.remove(((targets[targets >= 2]).index).values)
    D.normalize(37.28941975)
    D.shuffle()

    # sampler = StratifiedKFold(10, shuffle=False)

    Feats = D.asMatrix()[:, :6100]
    targets, targets_name = D.getMulticlassTargets()

    # idxs = None
    # for train_index, test_index in sampler.split(Feats, targets):
    #     if(idxs is None):
    #         idxs = test_index
    #     else:
    #         idxs = np.concatenate((idxs, test_index), axis=0)
    # Feats = Feats[idxs]
    # targets = targets[idxs]
    n = int(len(targets) * 0.8)
    X = Feats[:n]
    Y = targets[:n]
    Xtest = Feats[n:]
    Ytest = targets[n:]
    return ((X, Y), (Xtest, Ytest))


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


def main(save_file, Dtrain, Dtest=None):
    metrics = {'Accuracy': accuracy_score,
               'f-macro': lambda x, y: f1_score(x, y, average='macro'),
               'prec-macro': lambda x, y: precision_score(x, y, average='macro'),
               'recall-macro': lambda x, y: recall_score(x, y, average='macro')}
    X, Y = Dtrain

    net = TripletNetwork(net_arch=lmelloEmbeddingNet(8).cuda())
    # net = TripletNetwork.load('/tmp/tmp.pt', net_arch=lmelloEmbeddingNet(8).cuda(), map_location='cuda:0')

    # net = EmbeddingWrapper(num_outputs=8, net_arch=lmelloEmbeddingNet)
    # net = EmbeddingWrapper(num_outputs=8, net_arch=EmbeddingNetMNIST)
    # net.train(Dtrain, learning_rate=1e-3, num_subepochs=20, batch_size=25, niterations=1,
    #           loss_function_generator=DistanceCorrelationLoss)
    # net.train(Dtrain, learning_rate=1e-3, num_subepochs=8, batch_size=25, niterations=8,
    #           loss_function_generator=DistanceCorrelationLoss)
    # net.train(Dtrain, learning_rate=1e-3, num_subepochs=8, batch_size=25, num_epochs=1,
    #           loss_function_generator=OnlineTripletLoss)
    # net.train(Dtrain, learning_rate=1e-3, num_subepochs=9, batch_size=25, niterations=9,
    #       loss_function_generator=CorrelationMatrixLoss)
    # net.save(save_file)

    """Test"""
    if(Dtest is not None):
        Xtest, Ytest = Dtest
        clf = AugmentedClassifier(RandomForestClassifier(n_estimators=100),
                                  net_arch=net.net_arch)
        clf.train_tripletnet = False
        # clf.set_train_params(num_epochs=1,
        #                      num_subepochs=13,
        #                      batch_size=25,
        #                      learning_rate=1e-3)

        clf.fit(X, Y)
        preds = clf.predict(Xtest)
        stats = {name: [m(Ytest, preds)] for name, m in metrics.items()}
        df = pd.DataFrame(stats)
        print("")
        print(df)


def main2(save_file, D, Dtest=None, end2end=True):
    metrics = {'Accuracy': accuracy_score,
               'f-macro': lambda x, y: f1_score(x, y, average='macro'),
               'prec-macro': lambda x, y: precision_score(x, y, average='macro'),
               'recall-macro': lambda x, y: recall_score(x, y, average='macro')}

    nclasses = len(np.bincount(D[1]))
    if(end2end):
        encoder_net = lmelloEmbeddingNet(8)
    else:
        fpath = '/home/lhsmello/ufes/doutorado/TripletNet-on-ESP/saved_models/lossanalysis/12-05-2020_32outs.pt'
        encoder_net = EmbeddingWrapper.loadModel(fpath).embedding_net
        for p in encoder_net.parameters():
            p.requires_grad = False
    net = ClassifierConvNet(nclasses, encoder_net=encoder_net)

    X, Y = D
    net.fit(X, Y)

    """Test"""
    if(Dtest is not None):
        Xtest, Ytest = Dtest
        preds = net.predict(Xtest)
        stats = {name: [m(Ytest, preds)] for name, m in metrics.items()}
        df = pd.DataFrame(stats)
        print("")
        print(df)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, required=True)
    args = parser.parse_args()

    Dtrain, Dtest = loadRPDBCSData()
    # D = loadMNIST()
    main(args.outfile, Dtrain, Dtest)
    # main2(args.outfile, Dtrain, Dtest, end2end=True)
