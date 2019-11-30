import torch
import torch.optim as optim
from siamese_triplet.networks import ClassificationNet
from siamese_triplet.metrics import AverageNonzeroTripletsMetric, AccumulatedAccuracyMetric
from siamese_triplet.losses import OnlineTripletLoss
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
    optimizer = optim.Adam(model.parameters(), lr=9e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 500
    siamese_triplet.trainer.fit(train_loader, test_loader, model, loss_fn,
                                optimizer, scheduler, n_epochs, use_cuda, log_interval,
                                metrics=[AccumulatedAccuracyMetric()])
    return model


def train_tripletNetwork(triplet_train_loader, triplet_test_loader, model, triplet_selector, n_epochs=15, margin=0.5, use_cuda=True):
    if use_cuda:
        model.cuda()
    loss_fn = OnlineTripletLoss(margin, triplet_selector(margin))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1, last_epoch=-1)
    log_interval = 200
    siamese_triplet.trainer.fit(triplet_train_loader, triplet_test_loader, model, loss_fn,
                                optimizer, scheduler, n_epochs, use_cuda, log_interval,
                                metrics=[AverageNonzeroTripletsMetric()])
    return model


def main(save_file):
    from rpdbcs.datahandler.dataset import readDataset
    from datahandler import RPDBCSTorchDataset
    from siamese_triplet.datasets import BalancedBatchSampler
    from networks import lmelloEmbeddingNet
    # Strategies for selecting triplets within a minibatch
    from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector

    D = readDataset('data/data_classified_v4/freq.csv', 'data/data_classified_v4/labels.csv',
                    remove_first=100, nsigs=33000, npoints=20000)
    D.normalize(f_hz="min")
    D.shuffle()
    train_dataset = RPDBCSTorchDataset(D, train=True, signal_size=11028, holdout=0.9)
    test_dataset = RPDBCSTorchDataset(D, train=False, signal_size=11028,
                                      scaler=train_dataset.scaler, holdout=0.9)
    myclasses = train_dataset.targets_name
    assert(len(myclasses) == len(test_dataset.targets_name))
    # myclasses = {c: (myclasses[c], COLOR_CODES[myclasses[c]]) for c in myclasses}
    train_batch_sampler = BalancedBatchSampler(
        train_dataset.targets, n_classes=len(myclasses), n_samples=8)
    # test_batch_sampler = BalancedBatchSampler(test_dataset.targets, n_classes=len(myclasses), n_samples=8)
    ######################

    cuda = torch.cuda.is_available()

    # Set up data loaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # batch_size = 256
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    triplet_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    # triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    num_outputs = 8
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

    triplet_model = train_tripletNetwork(triplet_train_loader, None,
                                         lmelloEmbeddingNet(num_outputs),
                                         RandomNegativeTripletSelector, margin=0.5, use_cuda=cuda)
    torch.save({'state_dict': triplet_model.state_dict()}, save_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, required=True)
    args = parser.parse_args()

    main(args.outfile)
