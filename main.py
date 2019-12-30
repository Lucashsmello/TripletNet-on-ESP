from trainer import train_classifier, train_tripletNetwork
# Strategies for selecting triplets within a minibatch
from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
from networks import LarsHuyEmbeddingNet, lmelloEmbeddingNet, extract_embeddings
from classifiers.augmented_classifier import EmbeddingWrapper
from siamese_triplet.datasets import BalancedBatchSampler
import torch
from visualization import plot_embeddings, plot_embeddings3d, pairplot_embeddings
from rpdbcs.datahandler.dataset import readDataset
from rpdbcs.datahandler.dataview import COLOR_CODES
from datahandler import RPDBCSTorchDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--loadmodel', type=str)
args = parser.parse_args()

D = readDataset('data/data_classified_v6/freq.csv', 'data/data_classified_v6/labels.csv',
                remove_first=100, nsigs=2000, npoints=10800)
D.normalize(f_hz="min")
# D.shuffle()
train_dataset = RPDBCSTorchDataset(D, train=True, signal_size=6100, holdout=1.0)
test_dataset = RPDBCSTorchDataset(D, train=False, signal_size=6100,
                                  scaler=train_dataset.scaler, holdout=0.8)
myclasses = train_dataset.targets_name
assert(len(myclasses) == len(test_dataset.targets_name))
myclasses = {i: (c, COLOR_CODES[c]) for i, c in myclasses.items()}
train_batch_sampler = BalancedBatchSampler(
    train_dataset.targets, n_classes=len(myclasses), n_samples=8)
test_batch_sampler = BalancedBatchSampler(
    test_dataset.targets, n_classes=len(myclasses), n_samples=8)
######################


cuda = torch.cuda.is_available()

# Set up data loaders
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
batch_size = 256
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
triplet_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_sampler=train_batch_sampler, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_sampler=test_batch_sampler, **kwargs)

num_outputs = 2

triplet_model = lmelloEmbeddingNet(num_outputs)
if(cuda):
    triplet_model.cuda()

if(args.loadmodel):
    checkpoint = torch.load(args.loadmodel)
    triplet_model.load_state_dict(checkpoint['state_dict'])
else:
    for i in range(3):
        triplet_model = train_tripletNetwork(triplet_train_loader, None,
                                             triplet_model,
                                             RandomNegativeTripletSelector, margin=0.2, lr=4e-5, n_epochs=7, use_cuda=cuda)
        triplet_model = train_tripletNetwork(triplet_train_loader, None,
                                             triplet_model,
                                             HardestNegativeTripletSelector, margin=0.1, lr=4e-5, n_epochs=7, use_cuda=cuda)
train_embeddings_tl, train_labels_tl = extract_embeddings(
    train_loader, triplet_model, num_outputs=num_outputs, use_cuda=cuda)
val_embeddings_tl, val_labels_tl = extract_embeddings(
    test_loader, triplet_model, num_outputs=num_outputs, use_cuda=cuda)
pairplot_embeddings(train_embeddings_tl, train_labels_tl, myclasses)
pairplot_embeddings(val_embeddings_tl, val_labels_tl, myclasses)
# plot_embeddings(train_embeddings_tl, train_labels_tl, myclasses)
# plot_embeddings(val_embeddings_tl, val_labels_tl, myclasses)
