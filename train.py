import torch
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset
from sklearn.model_selection import StratifiedShuffleSplit
from tripletnet.classifiers.augmented_classifier import EmbeddingWrapper, ClassifierConvNet


def main(save_file):
    data_dir = 'data/data_classified_v6'
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=33000, npoints=10800)
    targets, _ = D.getMulticlassTargets()
    # D.remove((targets[targets == 4].index).values)
    D.normalize(37.28941975)
    D.shuffle()
    Feats = D.asMatrix()[:, :6100]
    targets, targets_name = D.getMulticlassTargets()
    print(targets_name)
    n = int(len(targets) * 0.8)
    Xtrain = Feats[:n]
    Ytrain = targets[:n]
    # sampler = StratifiedShuffleSplit(1, test_size=0.2, random_state=123)
    # for train_index, _ in sampler.split(Feats, targets):
    #     Xtrain = Feats[train_index]
    #     Ytrain = targets[train_index]
    #     break
    net = EmbeddingWrapper(num_outputs=8)
    net.train(Xtrain, Ytrain, learning_rate=1e-3, num_subepochs=15, batch_size=16)
    net.save(save_file)


def main2(save_file):
    encoder_net = EmbeddingWrapper.loadModel(
        '/home/lhsmello/ufes/doutorado/cogvisual2/saved_models/feature_analysis/8feats_tripletspace.pt').embedding_net
    for p in encoder_net.parameters():
        p.requires_grad = False
    data_dir = 'data/data_classified_v6'
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=33000, npoints=10800)
    targets, _ = D.getMulticlassTargets()
    D.normalize(37.28941975)
    D.shuffle()
    Feats = D.asMatrix()[:, :6100]
    targets, targets_name = D.getMulticlassTargets()
    print(targets_name)
    n = int(len(targets) * 0.8)
    Xtrain = Feats[:n]
    Ytrain = targets[:n]

    net = ClassifierConvNet(5, encoder_net=encoder_net)
    net.fit(Xtrain, Ytrain)
    net.save(save_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, required=True)
    args = parser.parse_args()

    main2(args.outfile)
