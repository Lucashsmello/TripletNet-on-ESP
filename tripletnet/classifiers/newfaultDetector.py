from classifiers.QDAExtended import QDAExtended
from classifiers.augmented_classifier import EmbeddingWrapper
from sklearn.base import BaseEstimator
import numpy as np


class NewFaultDetector(BaseEstimator):
    def __init__(self, out_code, base_dir=None, num_outputs: int = 8, threshold=0.001):
        self.base_dir = base_dir
        self.num_outputs = num_outputs
        self.out_code = out_code
        self.threshold=threshold

    def fit(self, X, Y):
        self.encodder = EmbeddingWrapper(self.base_dir, self.num_outputs)
        self.QDAext = QDAExtended()
        self.encodder.train(X, Y, learning_rate=1e-3, num_subepochs=16, batch_size=16, niterations=16)
        Xnew = self.encodder.embed(X)
        self.QDAext.fit(Xnew, Y)

    def predict(self, X, y=None):
        Xnew = self.encodder.embed(X)
        likes = self.QDAext.likelihood(Xnew)
        return np.array([self.out_code if max(l) < self.threshold else 0 for l in likes], np.int)
