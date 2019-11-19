from siamese_triplet.losses import OnlineTripletLoss
from siamese_triplet.metrics import AverageNonzeroTripletsMetric, AccumulatedAccuracyMetric
# Strategies for selecting triplets within a minibatch
from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
from siamese_triplet.networks import ClassificationNet
from networks import LarsHuyEmbeddingNet, lmelloEmbeddingNet
from siamese_triplet.TripletDataset import TripletDataset
from siamese_triplet.trainer import fit
from siamese_triplet.datasets import BalancedBatchSampler
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from visualization import plot_embeddings, plot_embeddings3d
from rpdbcs.datahandler.dataset import readDataset
from RPDBCSTorchDataset import RPDBCSTorchDataset


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for samples, target in dataloader:
            if cuda:
                samples = samples.cuda()
            embeddings[k:k+len(samples)] = model.get_embedding(samples).data.cpu().numpy()
            labels[k:k+len(samples)] = target.numpy()
            k += len(samples)
    return embeddings, labels
########################


#train_dataset = MyDataset('data/fakedata.csv', train=True)
#test_dataset = MyDataset('data/fakedata.csv', train=False)

D = readDataset('data/data-11-11-2019_2/freq.csv',
                'data/data-11-11-2019_2/labels.csv', remove_first=100, nsigs=25000, npoints=20000)
D.normalize(f_hz="min")
D.shuffle()
train_dataset = RPDBCSTorchDataset(D, train=True, signal_size=11028, holdout=0.85)
test_dataset = RPDBCSTorchDataset(D, train=False, signal_size=11028,
                                  scaler=train_dataset.scaler, holdout=0.85)
myclasses = train_dataset.targets_name
assert(len(myclasses) == len(test_dataset.targets_name))
train_batch_sampler = BalancedBatchSampler(
    train_dataset.targets, n_classes=len(myclasses), n_samples=8)
test_batch_sampler = BalancedBatchSampler(
    test_dataset.targets, n_classes=len(myclasses), n_samples=8)
######################

# Common setup

cuda = torch.cuda.is_available()


# Set up data loaders
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
batch_size = 256
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
#triplet_train_dataset = TripletDataset(train_dataset)
#triplet_test_dataset = TripletDataset(test_dataset)

# batch_size = 64
# triplet_train_loader = torch.utils.data.DataLoader(
#     triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# triplet_test_loader = torch.utils.data.DataLoader(
#     triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
triplet_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_sampler=train_batch_sampler, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters

embedding_net = lmelloEmbeddingNet()
#model = TripletNet(embedding_net)
model = embedding_net
if cuda:
    model.cuda()
# loss_fn = TripletLoss(margin)
margin = 1.
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 25
log_interval = 200
fit(triplet_train_loader, None, model, loss_fn,
    optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])

train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
plot_embeddings(train_embeddings_tl, train_labels_tl, myclasses)
plot_embeddings(val_embeddings_tl, val_labels_tl, myclasses)
