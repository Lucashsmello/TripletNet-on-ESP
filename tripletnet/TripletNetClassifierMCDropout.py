from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from tripletnet.networks import TripletNetwork
import numpy as np
import torch


class TripletNetClassifierMCDropout(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier, *args, mc_iters=20, cache_dir, **kwargs):
        super().__init__()
        self.tripletnet = TripletNetwork(*args, **kwargs)
        self.cache_dir = cache_dir
        if(base_classifier is not None):
            self.setBaseClassifier(base_classifier)
        self.mc_iters = mc_iters

    def setBaseClassifier(self, base_classifier):
        self.base_classifier = base_classifier
        if(self.cache_dir is None):
            self.pipeline = Pipeline([('transformer', self.tripletnet),
                                      ('base_classifier', base_classifier)])
        else:
            self.pipeline = Pipeline([('transformer', self.tripletnet),
                                      ('base_classifier', base_classifier)],
                                     memory=self.cache_dir)

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        preds = [self.pipeline.predict_proba_stochastic(X) for _ in range(self.mc_iters)]
        return np.mean(preds, axis=0)

    def get_params(self, deep=True, **kwargs):
        params = self.tripletnet.get_params(deep, **kwargs)
        params['mc_iters'] = self.mc_iters
        params['cache_dir'] = self.cache_dir
        params['base_classifier'] = self.base_classifier

        return params
