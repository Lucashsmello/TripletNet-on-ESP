import torch
import torch.optim as optim
from siamese_triplet.networks import ClassificationNet
from siamese_triplet.metrics import AverageNonzeroTripletsMetric, AccumulatedAccuracyMetric
from siamese_triplet.losses import OnlineTripletLoss
from siamese_triplet.datasets import BalancedBatchSampler
import siamese_triplet.trainer


def train_classifier(train_loader, test_loader, embedding_net, n_epochs=15, use_cuda=True):
    nclasses = train_loader.dataset.targets_name
    if(train_loader.dataset.targets_name is None):
        nclasses = 5  # FIXME
    else:
        nclasses = len(train_loader.dataset.targets_name)
    if(use_cuda):
        embedding_net.cuda()
    model = ClassificationNet(embedding_net, n_classes=nclasses)
    if use_cuda:
        model.cuda()
    loss_fn = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    log_interval = 500
    siamese_triplet.trainer.fit(train_loader, test_loader, model, loss_fn,
                                optimizer, scheduler, n_epochs, use_cuda, log_interval,
                                metrics=[AccumulatedAccuracyMetric()])
    return model


def train_tripletNetworkAdvanced(data_train, triplet_test_loader, model,
                                 triplet_strategies, niterations=1, gamma=0, beta=0,
                                 bootstrap_sample=0,
                                 use_cuda=True, network_model_plot=None, X_plot=None, Y_plot=None):
    if use_cuda:
        model.cuda()
    log_interval = 2000
    g = 1
    b = 1
    kwargs = {'num_workers': 1, 'pin_memory': True}

    for i in range(niterations):
        if(bootstrap_sample > 0):
            data_train.setBootstrap(bootstrap_sample)
        T = data_train.getTargets()
        batch_sampler = BalancedBatchSampler(T, n_classes=data_train.nclasses, n_samples=16)
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
            scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
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
    # from datahandler import RPDBCSTorchDataset
    # from siamese_triplet.datasets import BalancedBatchSampler
    # from networks import lmelloEmbeddingNet
    from classifiers.augmented_classifier import EmbeddingWrapper
    # Strategies for selecting triplets within a minibatch

    data_dir = 'data/data_classified_v6'
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=33000, npoints=10800)
    D.normalize(f_hz="min")
    D.shuffle()
    Feats = D.asMatrix()[:, :6100]
    targets, _ = D.getMulticlassTargets()
    n = int(len(targets) * 0.8)
    Feats = Feats[:n]
    targets = targets[:n]
    encodder_net = EmbeddingWrapper(None, 8)
    encodder_net.train(Feats, targets)
    torch.save({'state_dict': encodder_net.embedding_net.state_dict()}, save_file)


"""
def oldmain():
    train_dataset = RPDBCSTorchDataset(D, train=True, signal_size=6100, holdout=0.95)
    # test_dataset = RPDBCSTorchDataset(D, train=False, signal_size=11028,
    #                                  scaler=train_dataset.scaler, holdout=0.95)
    myclasses = train_dataset.targets_name
    # assert(len(myclasses) == len(test_dataset.targets_name))
    # myclasses = {c: (myclasses[c], COLOR_CODES[myclasses[c]]) for c in myclasses}
    train_batch_sampler = BalancedBatchSampler(
        train_dataset.targets, n_classes=len(myclasses), n_samples=8)
    # test_batch_sampler = BalancedBatchSampler(test_dataset.targets, n_classes=len(myclasses), n_samples=8)
    ######################

    # Set up data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True}
    # batch_size = 256
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    triplet_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    # triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    num_outputs = 2
    '''
    classifier_model = train_classifier(train_loader, test_loader,
                                        lmelloEmbeddingNet(num_outputs), use_cuda=cuda)
    train_embeddings_baseline, train_labels_baseline = extract_embeddings(
        train_loader, classifier_model, num_outputs=num_outputs)
    val_embeddings_baseline, val_labels_baseline = extract_embeddings(
        test_loader, classifier_model, num_outputs=num_outputs)
    plot_embeddings(train_embeddings_baseline, train_labels_baseline, myclasses)
    plot_embeddings(val_embeddings_baseline, val_labels_baseline, myclasses)
    '''
    triplet_model = lmelloEmbeddingNet(num_outputs)
    triplet_model.cuda()
    for i in range(3):
        triplet_model = train_tripletNetwork(triplet_train_loader, None,
                                             triplet_model,
                                             RandomNegativeTripletSelector, margin=0.2, lr=5e-5 * (0.99**i), n_epochs=7)
        triplet_model = train_tripletNetwork(triplet_train_loader, None,
                                             triplet_model,
                                             HardestNegativeTripletSelector, margin=0.1, lr=5e-5 * (0.99**i), n_epochs=7)
    torch.save({'state_dict': triplet_model.state_dict()}, save_file)
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, required=True)
    args = parser.parse_args()

    main(args.outfile)
