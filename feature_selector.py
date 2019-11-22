from sklearn.base import TransformerMixin

# def _ManualFeatures(X, y, manual_selection_list):
# return [1.0 if(i in manual_selection_list) else 0.0 for i in range(X.shape[1])]


class ManualFeaturesSelector(TransformerMixin):
    def __init__(self, manual_selection_list):
        self.manual_selection_list = manual_selection_list
        #super().__init__(lambda X, y: _ManualFeatures(X, y, manual_selection_list), len(manual_selection_list))

    def transform(self, X):
        return X[:, self.manual_selection_list]

    def fit(self, X=None, y=None):
        return self
