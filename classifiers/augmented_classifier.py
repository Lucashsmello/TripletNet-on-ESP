import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from datahandler import BasicTorchDataset, fitScaler
from siamese_triplet.datasets import BalancedBatchSampler
from siamese_triplet.utils import RandomNegativeTripletSelector, HardestNegativeTripletSelector, SemihardNegativeTripletSelector
from trainer import train_tripletNetwork, train_classifier
from networks import extract_embeddings, lmelloEmbeddingNet
import numpy as np
import os

from visualization import pairplot_embeddings  # REMOVEME


FOLD_ID = 0


def RESET_FOLD_ID():
    global FOLD_ID
    FOLD_ID = 0


# def _hashkey(name):
#    s = ";".join([str(x) for x in X[0]]).encode('utf-8')
#    return hashlib.sha256(s).hexdigest()


def _loadTorchModel(fpath, rawmodel):
    if(os.path.exists(fpath)):
        print("Loading model")
        checkpoint = torch.load(fpath)
        rawmodel.load_state_dict(checkpoint['state_dict'])
        rawmodel.cuda()
        return True
    return False


class ClassifierConvNet(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.scaler = fitScaler(X)
        X = self.scaler.transform(X)
        D = BasicTorchDataset(X, y)
        kwargs = {'num_workers': 1, 'pin_memory': True}
        dataloader = torch.utils.data.DataLoader(D, batch_size=16, **kwargs)
        self.netmodel = train_classifier(
            dataloader, None, lmelloEmbeddingNet(32), n_epochs=20, use_cuda=True)
        # Return the classifier
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        D = BasicTorchDataset(X, None)
        kwargs = {'num_workers': 1, 'pin_memory': True}
        dataloader = torch.utils.data.DataLoader(D, batch_size=128, **kwargs)
        preds = np.zeros(len(dataloader.dataset))
        k = 0
        with torch.no_grad():
            self.netmodel.eval()
            for samples, _ in dataloader:
                samples = samples.cuda()
                scores = self.netmodel(samples).data.cpu().numpy()
                preds[k:k+len(samples)] = [np.argmax(p) for p in scores]
                k += len(samples)
        return preds


class EmbeddingWrapper:
    def __init__(self, base_dir=None, num_outputs=8):
        self.fold_number = 1
        self.base_dir = base_dir
        self.num_outputs = num_outputs

    def train(self, X, y):
        global FOLD_ID
        FOLD_ID += 1
        self.scaler = fitScaler(X)

        self.embedding_net = lmelloEmbeddingNet(self.num_outputs)
        model_loaded = False

        if(self.base_dir is not None):
            if(self.base_dir[-3:] == '.pt' or self.base_dir[-4:] == '.pth'):
                fpath = self.base_dir
            else:
                fpath = "%s/%d-%d-%d-%d.pt" % (self.base_dir, FOLD_ID,
                                               X.shape[0], X.shape[1], sum(y))
            model_loaded = _loadTorchModel(fpath, self.embedding_net)

        if(not model_loaded):
            X = self.scaler.transform(X)
            print("Training new model")
            D = BasicTorchDataset(X, y)
            batch_sampler = BalancedBatchSampler(D.targets, n_classes=len(set(y)), n_samples=6)
            kwargs = {'num_workers': 1, 'pin_memory': True}
            dataloader = torch.utils.data.DataLoader(D, batch_sampler=batch_sampler, **kwargs)
            self.embedding_net = train_tripletNetwork(
                dataloader, None, self.embedding_net, RandomNegativeTripletSelector,
                n_epochs=18, margin=0.25, use_cuda=True)
            if(self.base_dir is not None):
                torch.save({'state_dict': self.embedding_net.state_dict()}, fpath)

    def embed(self, X):
        X = self.scaler.transform(X)
        D = BasicTorchDataset(X, None)
        kwargs = {'num_workers': 1, 'pin_memory': True}
        dataloader = torch.utils.data.DataLoader(D, batch_size=128, **kwargs)
        return extract_embeddings(dataloader, self.embedding_net, use_cuda=True, with_labels=False)


class AugmentedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classif, num_predefined_feats=0, savedir=None):
        self.embedding_net_wrapper = EmbeddingWrapper(savedir)
        self.base_classif = base_classif
        self.savedir = savedir
        self.num_predefined_feats = num_predefined_feats

    def get_params(self, deep=True):
        return {"base_classif": self.base_classif,
                "num_predefined_feats": self.num_predefined_feats,
                "savedir": self.savedir}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            # setattr(self, parameter, value)
            if(parameter == 'savedir'):
                self.savedir = value
                self.embedding_net_wrapper = EmbeddingWrapper(value)
            if(parameter == 'num_predefined_feats'):
                self.num_predefined_feats = value
            if(parameter == 'base_classif'):
                self.base_classif = value
        return self

    def _splitfeatures(self, X):
        X1 = X[:, :self.num_predefined_feats]
        X2 = X[:, self.num_predefined_feats:]
        return X1, X2

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X1, X2 = self._splitfeatures(X)
        self.classes_ = unique_labels(y)
        self.embedding_net_wrapper.train(X2, y)
        newX = self.embedding_net_wrapper.embed(X2)
        newX = np.concatenate((X1, newX), axis=1)
        #pairplot_embeddings(newX, y)
        self.base_classif.fit(newX, y)
        return self

    def predict(self, X):
        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        X1, X2 = self._splitfeatures(X)
        newX = self.embedding_net_wrapper.embed(X2)
        newX = np.concatenate((X1, newX), axis=1)
        return self.base_classif.predict(newX)
