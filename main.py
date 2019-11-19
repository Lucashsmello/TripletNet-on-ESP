from trainer import train_classifier, train_tripletNetwork
# Strategies for selecting triplets within a minibatch
from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
from networks import LarsHuyEmbeddingNet, lmelloEmbeddingNet, extract_embeddings
from siamese_triplet.datasets import BalancedBatchSampler
import torch
from visualization import plot_embeddings, plot_embeddings3d
from rpdbcs.datahandler.dataset import readDataset
from rpdbcs.datahandler.dataview import COLOR_CODES
from RPDBCSTorchDataset import RPDBCSTorchDataset


D = readDataset('data/data-11-11-2019_2/freq.csv', 'data/data-11-11-2019_2/labels.csv',
                remove_first=100, nsigs=25000, npoints=20000)
D.normalize(f_hz="min")
D.shuffle()
train_dataset = RPDBCSTorchDataset(D, train=True, signal_size=11028, holdout=0.7)
test_dataset = RPDBCSTorchDataset(D, train=False, signal_size=11028,
                                  scaler=train_dataset.scaler, holdout=0.7)
myclasses = train_dataset.targets_name
assert(len(myclasses) == len(test_dataset.targets_name))
myclasses = {c: (myclasses[c], COLOR_CODES[myclasses[c]]) for c in myclasses}
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

classifier_model = train_classifier(train_loader, test_loader, lmelloEmbeddingNet(), use_cuda=cuda)
train_embeddings_baseline, train_labels_baseline = extract_embeddings(
    train_loader, classifier_model)
val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, classifier_model)
plot_embeddings(train_embeddings_baseline, train_labels_baseline, myclasses)
plot_embeddings(val_embeddings_baseline, val_labels_baseline, myclasses)


triplet_model = train_tripletNetwork(triplet_train_loader, triplet_test_loader,
                                     lmelloEmbeddingNet(),
                                     RandomNegativeTripletSelector, margin=0.5, use_cuda=cuda)
train_embeddings_tl, train_labels_tl = extract_embeddings(
    train_loader, triplet_model, use_cuda=cuda)
val_embeddings_tl, val_labels_tl = extract_embeddings(
    test_loader, triplet_model, use_cuda=cuda)
plot_embeddings(train_embeddings_tl, train_labels_tl, myclasses)
plot_embeddings(val_embeddings_tl, val_labels_tl, myclasses)
