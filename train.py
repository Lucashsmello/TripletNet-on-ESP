import torch
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset
# from sklearn.model_selection import StratifiedShuffleSplit
from tripletnet.classifiers.augmented_classifier import EmbeddingWrapper, ClassifierConvNet
from tripletnet.losses import CorrelationMatrixLoss, DistanceCorrelationLoss
from siamese_triplet.losses import OnlineTripletLoss
from torchvision import transforms
from tripletnet.networks import EmbeddingNetMNIST, lmelloEmbeddingNet2, lmelloEmbeddingNet


def loadRPDBCSData(train=True):
    data_dir = 'data/data_classified_v6'
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=12000, npoints=10800)
    targets, _ = D.getMulticlassTargets()
    # D.remove((targets[targets == 4].index).values)
    D.normalize(37.28941975)
    D.shuffle()
    Feats = D.asMatrix()[:, :6100]
    targets, targets_name = D.getMulticlassTargets()
    n = int(len(targets) * 0.8)
    if(train):
        X = Feats[:n]
        Y = targets[:n]
    else:
        X = Feats[n:]
        Y = targets[n:]
    return X, Y


def loadMNIST(train=True):
    from torchvision.datasets import MNIST

    mean, std = 0.1307, 0.3081
    dataset = MNIST('/tmp', train=train, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((mean,), (std,))
                    ]))

    return dataset


def main(save_file, Dtrain):
    net = EmbeddingWrapper(num_outputs=32, net_arch=lmelloEmbeddingNet2)
    # net = EmbeddingWrapper(num_outputs=8, net_arch=EmbeddingNetMNIST)
    # net.train(Dtrain, learning_rate=1e-3, num_subepochs=20, batch_size=25, niterations=1,
    #           loss_function_generator=DistanceCorrelationLoss)
    # net.train(Dtrain, learning_rate=1e-3, num_subepochs=8, batch_size=25, niterations=8,
    #           loss_function_generator=DistanceCorrelationLoss)
    net.train(Dtrain, learning_rate=1e-3, num_subepochs=9, batch_size=25, niterations=9,
              loss_function_generator=OnlineTripletLoss)
    # net.train(Dtrain, learning_rate=1e-3, num_subepochs=9, batch_size=25, niterations=9,
    #       loss_function_generator=CorrelationMatrixLoss)
    net.save(save_file)


def main2(save_file, D):
    fpath = '/home/lhsmello/ufes/doutorado/TripletNet-on-ESP/saved_models/lossanalysis/12-05-2020_32outs_lincorr.pt'
    encoder_net = EmbeddingWrapper.loadModel(fpath).embedding_net
    for p in encoder_net.parameters():
        p.requires_grad = False

    X, Y = D

    net = ClassifierConvNet(5, encoder_net=encoder_net)
    net.fit(X, Y)
    net.save(save_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, required=True)
    args = parser.parse_args()

    # D = loadRPDBCSData()
    D = loadMNIST()
    main(args.outfile, D)
