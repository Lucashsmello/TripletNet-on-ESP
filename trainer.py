import torch
import torch.optim as optim
from siamese_triplet.networks import ClassificationNet
from siamese_triplet.metrics import AverageNonzeroTripletsMetric, AccumulatedAccuracyMetric
from siamese_triplet.losses import OnlineTripletLoss
from siamese_triplet.datasets import BalancedBatchSampler
import siamese_triplet.trainer


def train_classifier(train_loader, test_loader, embedding_net, n_epochs=15, use_cuda=True):
    nclasses = len(train_loader.dataset.targets.unique())
    if(use_cuda):
        embedding_net.cuda()
    model = ClassificationNet(embedding_net, n_classes=nclasses)
    if use_cuda:
        model.cuda()
    # loss_fn = torch.nn.NLLLoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 35, gamma=0.1, last_epoch=-1)
    log_interval = 50000
    siamese_triplet.trainer.fit(train_loader, test_loader, model, loss_fn,
                                optimizer, scheduler, n_epochs, use_cuda, log_interval,
                                metrics=[AccumulatedAccuracyMetric()])
    return model


def train_tripletNetworkAdvanced(data_train, triplet_test_loader, model,
                                 triplet_strategies, niterations=1, gamma=0, beta=0,
                                 batch_size=16,
                                 use_cuda=True, network_model_plot=None, X_plot=None, Y_plot=None):
    if use_cuda:
        model.cuda()
    log_interval = 2000
    g = 1
    b = 1
    kwargs = {'num_workers': 1, 'pin_memory': True}

    for i in range(niterations):
        T = data_train.getTargets()
        batch_sampler = BalancedBatchSampler(T, n_classes=data_train.nclasses, n_samples=batch_size)
        triplet_train_loader = torch.utils.data.DataLoader(
            data_train, batch_sampler=batch_sampler, **kwargs)

        print("====train_tripletNetworkAdvanced: iteration %d====" % (i+1))
        for j, strat in enumerate(triplet_strategies):
            lr = strat['learning-rate'] * g
            n_epochs = strat['nepochs']
            margin = strat['margin'] * b
            if('hard_factor' in strat):
                hard_factor = strat['hard_factor']
                triplet_sel = strat['triplet-selector'](margin, hard_factor)
            else:
                triplet_sel = strat['triplet-selector'](margin)
            loss_fn = OnlineTripletLoss(margin, triplet_sel)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
            siamese_triplet.trainer.fit(triplet_train_loader, triplet_test_loader, model, loss_fn,
                                        optimizer, scheduler, n_epochs, use_cuda, log_interval,
                                        metrics=[AverageNonzeroTripletsMetric()])

            if(network_model_plot):
                from visualization import pairplot_embeddings
                import matplotlib.pyplot as plt
                pairplot_embeddings(network_model_plot.embed(X_plot), Y_plot, show=False)
                plt.savefig("%s/%.2d-%.2d.png" % ("figs/tmp", i, j))
                plt.clf()
        g *= 1-gamma
        b *= 1-beta
    return model


def train_tripletNetwork(triplet_train_loader, triplet_test_loader, model, triplet_selector, n_epochs=15, margin=0.5, lr=1e-5, use_cuda=True):
    if use_cuda:
        model.cuda()
    loss_fn = OnlineTripletLoss(margin, triplet_selector(margin))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 2000
    siamese_triplet.trainer.fit(triplet_train_loader, triplet_test_loader, model, loss_fn,
                                optimizer, scheduler, n_epochs, use_cuda, log_interval,
                                metrics=[AverageNonzeroTripletsMetric()])
    return model


def main(save_file):
    from rpdbcs.datahandler.dataset import readDataset
    from sklearn.model_selection import StratifiedShuffleSplit
    from classifiers.augmented_classifier import EmbeddingWrapper, ClassifierConvNet

    data_dir = 'data/data_classified_v6'
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=33000, npoints=10800)
    D.normalize(f_hz="min")
    D.shuffle()
    Feats = D.asMatrix()[:, :6100]
    targets, _ = D.getMulticlassTargets()
    n = int(len(targets) * 0.8)
    Xtrain = Feats[:n]
    Ytrain = targets[:n]
    # sampler = StratifiedShuffleSplit(1, test_size=0.2, random_state=123)
    # for train_index, _ in sampler.split(Feats, targets):
    #     Xtrain = Feats[train_index]
    #     Ytrain = targets[train_index]
    #     break
    net = ClassifierConvNet(5)
    net.fit(Xtrain, Ytrain)
    # net = EmbeddingWrapper(None, 8)
    # net.train(Xtrain, Ytrain)
    net.save(save_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, required=True)
    args = parser.parse_args()

    main(args.outfile)
