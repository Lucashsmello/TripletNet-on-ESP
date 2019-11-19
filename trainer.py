import torch
import torch.optim as optim
from siamese_triplet.networks import ClassificationNet
from siamese_triplet.metrics import AverageNonzeroTripletsMetric, AccumulatedAccuracyMetric
from siamese_triplet.losses import OnlineTripletLoss
import siamese_triplet.trainer


def train_classifier(train_loader, test_loader, embedding_net, use_cuda=True):
    classes = train_loader.dataset.targets_name
    model = ClassificationNet(embedding_net, n_classes=len(classes))
    if use_cuda:
        model.cuda()
    loss_fn = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 25
    log_interval = 50
    siamese_triplet.trainer.fit(train_loader, test_loader, model, loss_fn,
                                optimizer, scheduler, n_epochs, use_cuda, log_interval,
                                metrics=[AccumulatedAccuracyMetric()])
    return model


def train_tripletNetwork(triplet_train_loader, triplet_test_loader, model, triplet_selector, margin=0.5, use_cuda=True):
    if use_cuda:
        model.cuda()
    loss_fn = OnlineTripletLoss(margin, triplet_selector(margin))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 25
    log_interval = 200
    siamese_triplet.trainer.fit(triplet_train_loader, None, model, loss_fn,
                                optimizer, scheduler, n_epochs, use_cuda, log_interval,
                                metrics=[AverageNonzeroTripletsMetric()])
    return model
