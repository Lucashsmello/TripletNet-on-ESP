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
from ..networks import extract_embeddings, lmelloEmbeddingNet, lmelloEmbeddingNet2, BrunaEmbeddingNet
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
        #print('Loading model "%s"' % fpath)
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


class EmbeddingWrapper:
    def __init__(self, base_dir=None, num_outputs=8, net_arch=lmelloEmbeddingNet):
        self.base_dir = base_dir
        self.num_outputs = num_outputs
        self.model_loaded = False
        self.embedding_net = net_arch(num_outputs)

    @staticmethod
    def loadModel(fpath: str, net_arch=lmelloEmbeddingNet) -> 'EmbeddingWrapper':
        checkpoint = torch.load(fpath)
        num_outputs = checkpoint['num_outputs']
        if('net_arch_name' in checkpoint):
            msg = "Network arch in %s is incompatible with %s. Try using parameter 'net_arch' from %s.loadModel() properly."
            msg = msg % (fpath, net_arch.__name__, EmbeddingWrapper.__name__)
            assert(checkpoint['net_arch_name'] == net_arch.__name__), msg
        model = EmbeddingWrapper(num_outputs=num_outputs, net_arch=net_arch)
        model.embedding_net.load_state_dict(checkpoint['state_dict'])
        model.embedding_net.cuda()
        return model

    def save(self, fpath):
        data_to_save = {'state_dict': self.embedding_net.state_dict(),
                        'num_outputs': self.num_outputs,
                        'net_arch_name': self.embedding_net.__class__.__name__}
        torch.save(data_to_save, fpath)

    def train(self, D, learning_rate, num_subepochs, batch_size, niterations=16,
              loss_function_generator=OnlineTripletLoss, custom_trainepoch=train_epoch):
        if(not isinstance(D, torch.utils.data.Dataset)):
            D = BasicTorchDataset(D[0], D[1])
        margin1 = 1.0
        triplet_train_config = [
            {'triplet-selector': RandomNegativeTripletSelector,
             'learning-rate': learning_rate,
             'margin': margin1,
             'nepochs': num_subepochs
             }
        ]
        self.embedding_net = train_tripletNetworkAdvanced(
            D, None, self.embedding_net, triplet_train_config,
            gamma=0.1, beta=0.25, niterations=niterations, batch_size=batch_size,
            loss_function_generator=loss_function_generator, custom_trainepoch=custom_trainepoch)

    def _train_cached(self, X, y, learning_rate, num_subepochs, batch_size, niterations=16,
                      loss_function_generator=OnlineTripletLoss, custom_trainepoch=train_epoch):
        if(self.model_loaded):
            return
        global FOLD_ID
        FOLD_ID += 1
        # self.scaler = fitScaler(X)

        model_loaded = False

        if(self.base_dir is not None):
            if(self.base_dir[-3:] == '.pt' or self.base_dir[-4:] == '.pth'):
                fpath = self.base_dir
            else:
                # fpath = "%s/%d-%d-%d-%d-%d.pt" % (self.base_dir, FOLD_ID,
                #                                   X.shape[0], X.shape[1], sum(y), self.num_outputs)
                fpath = "%s/%d-%d-%d-%d-%d-%d-%d-%d.pt" % (self.base_dir, FOLD_ID,
                                                           X.shape[0], X.shape[1],
                                                           sum(y), self.num_outputs,
                                                           1e+6 * learning_rate, num_subepochs, batch_size)
            model_loaded = _loadTorchModel(fpath, self.embedding_net)
            if(not model_loaded):
                print('"%s" not found or not a torch model' % fpath)

        if(not model_loaded):
            print("Training new model")
            self.train((X, y), learning_rate, num_subepochs, batch_size=batch_size,
                       niterations=niterations,
                       loss_function_generator=loss_function_generator, custom_trainepoch=custom_trainepoch)
            if(self.base_dir is not None):
                self.save(fpath)

    def embed(self, X):
        """
        Transform features from the original to the triplet-space.

        Args:
            X (Matrix): Matrix of amplitudes.

        Returns:
            The encodding in an matrix of len(X) lines and self.num_outputs columns.
        """
        # X = self.scaler.transform(X)
        if(not isinstance(X, torch.utils.data.Dataset)):
            D = BasicTorchDataset(X, None)
        else:
            D = X
        kwargs = {'num_workers': 3, 'pin_memory': True}
        dataloader = torch.utils.data.DataLoader(D, batch_size=32, **kwargs)
        return extract_embeddings(dataloader, self.embedding_net, num_outputs=self.num_outputs, use_cuda=True, with_labels=False)


class AugmentedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classif, savedir=None,
                 learning_rate=1e-3, num_subepochs=35, num_epochs=16, batch_size=16,
                 custom_trainepoch=train_epoch, net_arch=lmelloEmbeddingNet):
        self.embedding_net_wrapper = EmbeddingWrapper(savedir, num_outputs=8, net_arch=lmelloEmbeddingNet)
        self.base_classif = base_classif
        self.savedir = savedir
        self.learning_rate = learning_rate
        self.num_subepochs = num_subepochs
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.custom_trainepoch = custom_trainepoch

    def get_params(self, deep=True):
        return {"base_classif": self.base_classif,
                "savedir": self.savedir,
                "learning_rate": self.learning_rate,
                "num_subepochs": self.num_subepochs,
                "batch_size": self.batch_size
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            # setattr(self, parameter, value)
            if(parameter == 'savedir'):
                self.savedir = value
                self.embedding_net_wrapper = EmbeddingWrapper(value)
            elif(parameter == 'base_classif'):
                self.base_classif = value
            elif(parameter == 'learning_rate'):
                self.learning_rate = value
            elif(parameter == 'num_subepochs'):
                self.num_subepochs = value
            elif(parameter == 'batch_size'):
                self.batch_size = value
        return self

    def fit(self, X, y):
        # X, y = check_X_y(X, y)
        # self.classes_ = unique_labels(y)
        self.embedding_net_wrapper._train_cached(X, y, self.learning_rate,
                                                 self.num_subepochs, self.batch_size, niterations=self.num_epochs,
                                                 custom_trainepoch=self.custom_trainepoch)
        newX = self.embedding_net_wrapper.embed(X)
        # pairplot_embeddings(newX, y)
        self.base_classif.fit(newX, y)
        return self

    def predict(self, X):
        X = check_array(X)
        newX = self.embedding_net_wrapper.embed(X)
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
