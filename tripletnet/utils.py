from sklearn.pipeline import Pipeline


class PipelineExtended(Pipeline):
    """
    Class made to work with :class:`~tripletnet.TripletNetClassifierMCDropout.TripletNetClassifierMCDropout`
    """

    def predict_proba_stochastic(self, X):
        n = 0
        for _, _, transform in self._iter(with_final=False):
            if(hasattr(transform, 'transform_stochastic')):
                X = transform.transform_stochastic(X)
                n += 1
            else:
                X = transform.transform(X)
        assert(n >= 1)
        return self._final_estimator.predict_proba(X)
