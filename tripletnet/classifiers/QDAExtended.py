from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats import multivariate_normal


class QDAExtended(QuadraticDiscriminantAnalysis):
    def __init__(self):
        super().__init__(store_covariance=True)

    def likelihood(self, X):
        RVs = [multivariate_normal(mean, covar, allow_singular=True)
               for mean, covar in zip(self.means_, self.covariance_)]

        modes = [rv.pdf(mean) for mean, rv in zip(self.means_, RVs)]
        return [[rv.pdf(x) / mode for mode, rv in zip(modes, RVs)] for x in X]
