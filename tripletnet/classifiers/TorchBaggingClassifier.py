from sklearn.ensemble import BaggingClassifier


class TorchBaggingClassifier(BaggingClassifier):
    def fit(self, X, y):
        X = X.reshape(X.shape[0], X.shape[2])
        super().fit(X, y)
        return self

    def predict(self, X):
        if(len(X.shape) == 3):
            X = X.reshape(X.shape[0], X.shape[2])
        return super().predict(X)

    def predict_proba(self, X):
        if(len(X.shape) == 3):
            X = X.reshape(X.shape[0], X.shape[2])
        return super().predict_proba(X)
