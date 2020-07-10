import torch
from torch import nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from ..datahandler import BasicTorchDataset
from siamese_triplet.networks import ClassificationNet
from siamese_triplet.utils import RandomNegativeTripletSelector, HardestNegativeTripletSelector, SemihardNegativeTripletSelector, HardNegativeTripletSelector
from ..trainer import train_tripletNetworkAdvanced, train_classifier, trainClassifier2
from ..networks import extract_embeddings, lmelloEmbeddingNet, lmelloEmbeddingNet2, TripletNetwork
import numpy as np
import os
from siamese_triplet.losses import OnlineTripletLoss
from siamese_triplet.trainer import train_epoch

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
        # print('Loading model "%s"' % fpath)
        checkpoint = torch.load(fpath)
        rawmodel.load_state_dict(checkpoint['state_dict'])
        rawmodel.cuda()
        return True
    return False


class ClassifierConvNet(BaseEstimator, ClassifierMixin):
    def __init__(self, nclasses: int, base_dir=None, encoder_net=None):
        self.base_dir = base_dir
        self.nclasses = nclasses
        if(encoder_net is None):
            self.num_outputs = 8
        else:
            for last_module in encoder_net.modules():
                pass
            self.num_outputs = last_module.out_features
        self.model_loaded = False
        self.learning_rate = 1e-3
        self.batch_size = 256
        self.num_steps_decay = 8
        self.embedding_net = encoder_net

    def load(self, fpath):
        """
        DEPRECATED
        """
        self.embedding_net = lmelloEmbeddingNet(self.num_outputs)
        self.netmodel = ClassificationNet(self.embedding_net, n_classes=self.nclasses)
        self.model_loaded = _loadTorchModel(fpath, self.netmodel)
        if(not self.model_loaded):
            print('"%s" not found or not a torch model' % fpath)

    @staticmethod
    def loadModel(fpath: str) -> 'ClassifierConvNet':
        checkpoint = torch.load(fpath)
        num_outputs = checkpoint['num_outputs']
        encoder_net = lmelloEmbeddingNet(num_outputs)
        convnet = ClassifierConvNet(nclasses=5, encoder_net=encoder_net)
        convnet.netmodel = ClassificationNet(encoder_net, n_classes=5)
        convnet.netmodel.load_state_dict(checkpoint['state_dict'])
        convnet.netmodel.cuda()

        return convnet

    def save(self, fpath):
        data_to_save = {'state_dict': self.netmodel.state_dict(),
                        'num_outputs': self.num_outputs}
        torch.save(data_to_save, fpath)

    def fit(self, X, y):
        global FOLD_ID
        FOLD_ID += 1
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        assert(len(self.classes_) ==
               self.nclasses), "Number of classes in training data should be %d" % self.nclasses
        # self.scaler = fitScaler(X)
        # X = self.scaler.transform(X)
        D = BasicTorchDataset(X, y)
        if(self.embedding_net is None):
            self.embedding_net = lmelloEmbeddingNet(num_outputs=self.num_outputs)
            if(self.base_dir is not None):
                if(self.base_dir[-3:] == '.pt' or self.base_dir[-4:] == '.pth'):
                    fpath = self.base_dir
                else:
                    fpath = "%s/%d-%d-%d-%d-%d-%d-%d-%d.pt" % (self.base_dir, FOLD_ID,
                                                               X.shape[0], X.shape[1], sum(
                                                                   y), self.num_outputs,
                                                               1e+6 * self.learning_rate, self.num_steps_decay, self.batch_size)
                self.netmodel = ClassificationNet(self.embedding_net, n_classes=self.nclasses)

                self.model_loaded = _loadTorchModel(fpath, self.netmodel)

        if(not self.model_loaded):
            class_sample_count = np.bincount(y)
            weights = torch.FloatTensor([1.0/class_sample_count[l] for l in y])
            # weights = torch.ones(len(y), dtype=torch.float32).cuda()
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(y))
            kwargs = {'num_workers': 4, 'pin_memory': True}
            dataloader = torch.utils.data.DataLoader(
                D, shuffle=False, sampler=sampler, batch_size=self.batch_size, **kwargs)
            # trainClassifier2(dataloader, self.embedding_net, lr=self.learning_rate, n_epochs=200)
            self.netmodel = train_classifier(
                dataloader, None, self.embedding_net,
                lr=self.learning_rate, num_steps_decay=self.num_steps_decay,  n_epochs=100)
            # Return the classifier
            if(self.base_dir is not None):
                self.save(fpath)
        return self

    def predict(self, X):
        # X = self.scaler.transform(X)
        D = BasicTorchDataset(X, None)
        kwargs = {'num_workers': 3, 'pin_memory': True}
        dataloader = torch.utils.data.DataLoader(D, batch_size=32, **kwargs)
        preds = np.empty(len(dataloader.dataset), dtype=np.int)
        k = 0
        with torch.no_grad():
            self.netmodel.eval()
            for samples, _ in dataloader:
                samples = samples.cuda()
                scores = self.netmodel(samples).data.cpu().numpy()
                preds[k:k+len(samples)] = [np.argmax(p) for p in scores]
                k += len(samples)
        return preds

    def embed(self, X):
        # X = self.scaler.transform(X)
        D = BasicTorchDataset(X, None)
        kwargs = {'num_workers': 3, 'pin_memory': True}
        dataloader = torch.utils.data.DataLoader(D, batch_size=32, **kwargs)
        return extract_embeddings(dataloader, self.embedding_net, use_cuda=True, with_labels=False)


class AugmentedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classif, net_arch):
        self.net_arch = net_arch
        self.tripletnet = None
        self.base_classif = base_classif
        self.train_tripletnet = True
        self.learning_rate = 1e-3
        self.num_subepochs = 10
        self.num_epochs = 10
        self.batch_size = 32
        self.custom_trainepoch = train_epoch

    def set_train_params(self, learning_rate, num_subepochs, num_epochs, batch_size):
        self.learning_rate = learning_rate
        self.num_subepochs = num_subepochs
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def set_custom_training(self, custom_trainepoch):
        self.custom_trainepoch = custom_trainepoch

    def get_params(self, deep=True):
        return {"base_classif": self.base_classif,
                "learning_rate": self.learning_rate,
                "num_subepochs": self.num_subepochs,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "custom_trainepoch": self.custom_trainepoch,
                "net_arch": self.net_arch,
                "train_tripletnet": self.train_tripletnet
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            # setattr(self, parameter, value)
            if(parameter == 'base_classif'):
                self.base_classif = value
            elif(parameter == 'learning_rate'):
                self.learning_rate = value
            elif(parameter == 'num_subepochs'):
                self.num_subepochs = value
            elif(parameter == 'batch_size'):
                self.batch_size = value
            elif(parameter == 'num_epochs'):
                self.num_epochs = value
            elif(parameter == 'net_arch'):
                self.net_arch = value
            elif(parameter == 'train_tripletnet'):
                self.train_tripletnet = value
        return self

    def fit(self, X, y):
        if(self.tripletnet is None):
            self.tripletnet = TripletNetwork(self.net_arch)
        # self.classes_ = unique_labels(y)
        if(self.train_tripletnet):
            self.tripletnet.train((X, y), self.learning_rate,
                                  self.num_subepochs, self.batch_size, num_epochs=self.num_epochs,
                                  custom_trainepoch=self.custom_trainepoch)
        newX = self.tripletnet.embed(X).cpu().numpy()
        # pairplot_embeddings(newX, y)
        self.base_classif.fit(newX, y)
        return self

    def predict(self, X):
        newX = self.tripletnet.embed(X).cpu().numpy()
        return self.base_classif.predict(newX)

    # def predict_proba(self, X):
    #     X1, X2 = self._splitfeatures(X)
    #     newX = self.embedding_net_wrapper.embed(X2)
    #     newX = np.concatenate((X1, newX), axis=1)
    #     return self.base_classif.predict_proba(newX)

    def destroy(self):
        del self.embedding_net_wrapper
        del self.base_classif


class ClassifierConvNet2(ClassifierConvNet):
    def __init__(self, nclasses: int, base_classif, base_dir=None,
                 learning_rate=1e-3, num_steps_decay=35, batch_size=500):
        super().__init__(nclasses, base_dir)
        self.augclassif = AugmentedClassifier(base_classif)
        self.learning_rate = learning_rate
        self.num_steps_decay = num_steps_decay
        self.batch_size = batch_size

    def fit(self, X, y):
        super().fit(X, y)
        self.augclassif.embedding_net_wrapper.embedding_net = self.embedding_net
        self.augclassif.embedding_net_wrapper.model_loaded = True
        self.augclassif.fit(X, y)

    def predict(self, X):
        return self.augclassif.predict(X)

    def get_params(self, deep=True):
        return {'nclasses': self.nclasses,
                'base_classif': self.augclassif.base_classif,
                'base_dir': self.base_dir,
                "learning_rate": self.learning_rate,
                "num_steps_decay": self.num_steps_decay,
                "batch_size": self.batch_size
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if(parameter == 'nclasses'):
                self.nclasses = value
            if(parameter == 'base_dir'):
                self.base_dir = value
                self.augclassif.savedir = value
                self.augclassif.embedding_net_wrapper = EmbeddingWrapper(value)
            if(parameter == 'base_classif'):
                self.augclassif.base_classif = value
            elif(parameter == 'learning_rate'):
                self.learning_rate = value
            elif(parameter == 'num_subepochs'):
                self.num_subepochs = value
            elif(parameter == 'batch_size'):
                self.batch_size = value
        return self
