import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from datahandler import BasicTorchDataset
from siamese_triplet.datasets import BalancedBatchSampler
from siamese_triplet.networks import ClassificationNet
from siamese_triplet.utils import RandomNegativeTripletSelector, HardestNegativeTripletSelector, SemihardNegativeTripletSelector, HardNegativeTripletSelector
from trainer import train_tripletNetworkAdvanced, train_classifier
from networks import extract_embeddings, lmelloEmbeddingNet, lmelloEmbeddingNet2, BrunaEmbeddingNet
import numpy as np
import os

from visualization import pairplot_embeddings  # REMOVEME
#from samplers import StratifiedSampler


FOLD_ID = 0


def timeit(func):
    import functools
    import time
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
    return newfunc


def RESET_FOLD_ID():
    global FOLD_ID
    FOLD_ID = 0


def INCREMENT_FOLD_ID():
    global FOLD_ID
    FOLD_ID += 1


# def _hashkey(name):
#    s = ";".join([str(x) for x in X[0]]).encode('utf-8')
#    return hashlib.sha256(s).hexdigest()

def _loadTorchModel(fpath, rawmodel):
    if(os.path.exists(fpath)):
        print('Loading model "%s"' % fpath)
        checkpoint = torch.load(fpath)
        rawmodel.load_state_dict(checkpoint['state_dict'])
        rawmodel.cuda()
        return True
    return False


class ClassifierConvNet(BaseEstimator, ClassifierMixin):
    def __init__(self, nclasses, base_dir=None):
        self.base_dir = base_dir
        self.nclasses = nclasses

    def fit(self, X, y):
        global FOLD_ID
        FOLD_ID += 1
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        # self.scaler = fitScaler(X)
        X = self.scaler.transform(X)
        D = BasicTorchDataset(X, y)
        embedding_net = BrunaEmbeddingNet()
        if(self.base_dir is not None):
            if(self.base_dir[-3:] == '.pt' or self.base_dir[-4:] == '.pth'):
                fpath = self.base_dir
            else:
                fpath = "%s/%d-%d-%d-%d.pt" % (self.base_dir, FOLD_ID,
                                               X.shape[0], X.shape[1], sum(y))
            self.netmodel = ClassificationNet(embedding_net, n_classes=self.nclasses)

            model_loaded = _loadTorchModel(fpath, self.netmodel)

        if(not model_loaded):
            class_sample_count = np.bincount(y)
            weights = torch.FloatTensor([1.0/class_sample_count[l] for l in y])
            weights.cuda()
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 500)
            kwargs = {'num_workers': 1, 'pin_memory': True}
            dataloader = torch.utils.data.DataLoader(D, shuffle=False, sampler=sampler, **kwargs)
            self.netmodel = train_classifier(
                dataloader, None, embedding_net, n_epochs=300, use_cuda=True)
            # Return the classifier
            if(self.base_dir is not None):
                torch.save({'state_dict': self.netmodel.state_dict()}, fpath)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        D = BasicTorchDataset(X, None)
        kwargs = {'num_workers': 1, 'pin_memory': True}
        dataloader = torch.utils.data.DataLoader(D, batch_size=32, **kwargs)
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

    def load(self, fpath):
        self.embedding_net = lmelloEmbeddingNet(self.num_outputs, num_knownfeats=0)
        model_loaded = _loadTorchModel(fpath, self.embedding_net)
        if(not model_loaded):
            print('"%s" not found or not a torch model' % fpath)

    def train(self, X, y):
        global FOLD_ID
        FOLD_ID += 1
        # self.scaler = fitScaler(X)

        self.embedding_net = lmelloEmbeddingNet(self.num_outputs, num_knownfeats=0)
        # self.embedding_net = BrunaEmbeddingNet(self.num_outputs, num_knownfeats=0)
        model_loaded = False

        if(self.base_dir is not None):
            if(self.base_dir[-3:] == '.pt' or self.base_dir[-4:] == '.pth'):
                fpath = self.base_dir
            else:
                fpath = "%s/%d-%d-%d-%d-%d.pt" % (self.base_dir, FOLD_ID,
                                                  X.shape[0], X.shape[1], sum(y), self.num_outputs)
            model_loaded = _loadTorchModel(fpath, self.embedding_net)
            if(not model_loaded):
                print('"%s" not found or not a torch model' % fpath)

        if(not model_loaded):
            # X = self.scaler.transform(X)
            print("Training new model")
            D = BasicTorchDataset(X, y)
            # batch_sampler = BalancedBatchSampler(D.targets, n_classes=len(set(y)), n_samples=8)
            # kwargs = {'num_workers': 1, 'pin_memory': True}
            # dataloader = torch.utils.data.DataLoader(D, batch_sampler=batch_sampler, **kwargs)
            margin1 = 1.0
            triplet_train_config = [
                # {'triplet-selector': SemihardNegativeTripletSelector,
                #  'learning-rate': 5e-4,
                #  'margin': margin1,
                #  'hard_factor': 1,
                #  'nepochs': 6
                #  },
                # {'triplet-selector': HardNegativeTripletSelector,
                #  'learning-rate': 5e-4,
                #  'margin': margin1+0.001,
                #  'nepochs': 6
                #  }
                {'triplet-selector': RandomNegativeTripletSelector,
                 'learning-rate': 1e-3,
                 'margin': margin1,
                 'nepochs': 30
                 }
            ]
            self.embedding_net = train_tripletNetworkAdvanced(
                D, None, self.embedding_net, triplet_train_config,
                gamma=0.1, beta=0.25, niterations=15, bootstrap_sample=0)
            if(self.base_dir is not None):
                torch.save({'state_dict': self.embedding_net.state_dict()}, fpath)

    def embed(self, X):
        # X = self.scaler.transform(X)
        D = BasicTorchDataset(X, None)
        kwargs = {'num_workers': 1, 'pin_memory': True}
        dataloader = torch.utils.data.DataLoader(D, batch_size=32, **kwargs)
        return extract_embeddings(dataloader, self.embedding_net, use_cuda=True, with_labels=False)


class AugmentedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classif, num_predefined_feats=0, savedir=None):
        self.embedding_net_wrapper = EmbeddingWrapper(savedir, num_outputs=8)
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
        # self.classes_ = unique_labels(y)
        self.embedding_net_wrapper.train(X2, y)
        newX = self.embedding_net_wrapper.embed(X2)
        newX = np.concatenate((X1, newX), axis=1)
        # pairplot_embeddings(newX, y)
        self.base_classif.fit(newX, y)
        return self

    # def fit(self, X, y):
    #     self.scaler = StandardScaler()
    #     X, y = check_X_y(X, y)
    #     X1 = X[:, :8]
    #     X2 = X[:, 8:]
    #     X1 = self.scaler.fit_transform(X1)
    #     X = np.concatenate((X1, X2), axis=1)
    #     self.embedding_net_wrapper.train(X, y)
    #     newX = self.embedding_net_wrapper.embed(X)
    #     # newX = np.concatenate((X, newX), axis=1)
    #     # pairplot_embeddings(newX, y)
    #     self.base_classif.fit(newX, y)
    #     return self

    def predict(self, X):
        X = check_array(X)
        X1, X2 = self._splitfeatures(X)
        newX = self.embedding_net_wrapper.embed(X2)
        newX = np.concatenate((X1, newX), axis=1)
        return self.base_classif.predict(newX)

    # def predict(self, X):
    #     X = check_array(X)
    #     X1 = X[:, :8]
    #     X2 = X[:, 8:]
    #     X1 = self.scaler.transform(X1)
    #     X = np.concatenate((X1, X2), axis=1)
    #     newX = self.embedding_net_wrapper.embed(X)
    #     return self.base_classif.predict(newX)

    # def predict_proba(self, X):
    #     X1, X2 = self._splitfeatures(X)
    #     newX = self.embedding_net_wrapper.embed(X2)
    #     newX = np.concatenate((X1, newX), axis=1)
    #     return self.base_classif.predict_proba(newX)

    def destroy(self):
        del self.embedding_net_wrapper
        del self.base_classif
