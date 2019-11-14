from siamese_triplet.losses import TripletLoss
from siamese_triplet.networks import TripletNet
from networks import LarsHuyEmbeddingNet, lmelloEmbeddingNet
from siamese_triplet.TripletDataset import TripletDataset
from siamese_triplet.trainer import fit
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from visualization import plot_embeddings
from rpdbcs.datahandler.dataset import readDataset
from RPDBCSTorchDataset import RPDBCSTorchDataset

#train_dataset = MyDataset('data/fakedata.csv', train=True)
#test_dataset = MyDataset('data/fakedata.csv', train=False)

D = readDataset('data/data-11-11-2019/freq.csv',
                'data/data-11-11-2019/labels.csv', remove_first=100, nsigs=25000, npoints=20000)
D.normalize(f_hz="min")
train_dataset = RPDBCSTorchDataset(D, train=True, signal_size=11028)
test_dataset = RPDBCSTorchDataset(D, train=False, signal_size=11028)
######################

# Common setup

cuda = torch.cuda.is_available()

myclasses = train_dataset.targets_name


# Set up data loaders
batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


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


# Set up data loaders

triplet_train_dataset = TripletDataset(train_dataset)
triplet_test_dataset = TripletDataset(test_dataset)
batch_size = 64
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(
    triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(
    triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters

margin = 1.
embedding_net = lmelloEmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 25
log_interval = 100
fit(triplet_train_loader, triplet_test_loader, model, loss_fn,
    optimizer, scheduler, n_epochs, cuda, log_interval)

train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_tl, train_labels_tl, myclasses)
# val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_tl, val_labels_tl)
