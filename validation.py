from joblib import Memory
from sklearn.base import TransformerMixin, BaseEstimator
import torch
from torch import nn
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset, getICTAI2016FeaturesNames
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import *
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._split import _BaseKFold
from tripletnet.classifiers.GridSearchCV_norefit import GridSearchCV_norefit
from rpdbcs.model_selection import StratifiedGroupKFold, rpdbcsKFold, StratifiedGroupSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
import skorch
from tripletnet.networks import TripletNetwork, TripletEnsembleNetwork
from tripletnet.datahandler import BalancedDataLoader
from torchvision import transforms
from tripletnet.networks import lmelloEmbeddingNet
import numpy as np
import pandas as pd
from tripletnet.datahandler import BasicTorchDataset
from tripletnet.callbacks import LoadEndState
import itertools
from tempfile import mkdtemp
from shutil import rmtree

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def loadRPDBCSData(data_dir='data/data_classified_v6', nsigs=100000):
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=nsigs, npoints=10800, dtype=np.float32)
    targets, _ = D.getMulticlassTargets()
    # print("WARNING: removing desalinhamento!")
    # D.remove(np.where(targets == 3)[0])  # removes desalinhamento
    print("Dataset length", len(D))
    D.normalize(37.28941975)
    D.shuffle()

    return D


def getBaseClassifiers(pre_pipeline=None):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    clfs = []
    knn = KNeighborsClassifier()
    knn_param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_param_grid = {'n_estimators': [100, 1000],
                     'max_features': [2, 3, 4, 5]}
    dtree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    # dtree = GridSearchCV(dtree,
    #                      {
    #                          'max_leaf_nodes': [10, 100],
    #                          'max_depth': [3, 6, 9, 12, 15]
    #                      }
    #                      )
    qda = QuadraticDiscriminantAnalysis()
    qda_param_grid = {'reg_param': [0.0, 1e-6, 1e-5]}
    clfs.append(("knn", knn, knn_param_grid))
    clfs.append(("DT", dtree, {}))
    clfs.append(("RF", rf, rf_param_grid))
    clfs.append(("NB", GaussianNB(), {}))
    clfs.append(("QDA", qda, qda_param_grid))

    if(pre_pipeline is not None):
        return [(cname, Pipeline([pre_pipeline, ('base_clf', c)]), {"base_clf__%s" % k: v for k, v in pgrid.items()})
                for cname, c, pgrid in clfs]

    return clfs


DEEP_CACHE_DIR = mkdtemp()
PIPELINE_CACHE_DIR = mkdtemp()


def getDeepTransformers():
    global DEEP_CACHE_DIR

    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=DEEP_CACHE_DIR, monitor='train_loss_best')
    parameters = {
        'callbacks': [checkpoint_callback, LoadEndState(checkpoint_callback)],
        'max_epochs': 100,
        'optimizer__lr': 1e-4,
        'criterion': TripletNetwork.OnlineTripletLossWrapper,
        'margin_decay_value': 0.75}
    deep_transf = []
    tripletnet = TripletNetwork(lmelloEmbeddingNet,
                                optimizer=optim.Adam, optimizer__weight_decay=1e-4,
                                module__num_outputs=8,
                                device='cuda',
                                train_split=None,
                                iterator_train=BalancedDataLoader, iterator_train__num_workers=0, iterator_train__pin_memory=False,
                                **parameters
                                )
    tripletnet_param_grid = {'batch_size': [40, 80],
                             'margin_decay_delay': [30, 40]}

    tripletnet_ensemble = TripletEnsembleNetwork(lmelloEmbeddingNet,
                                                 optimizer=optim.Adam, optimizer__weight_decay=1e-4,
                                                 module__num_outputs=32, device='cuda',
                                                 batch_size=80, margin_decay_delay=40,
                                                 train_split=None,
                                                 iterator_train=BalancedDataLoader, iterator_train__num_workers=0, iterator_train__pin_memory=False,
                                                 **parameters)
    tripletnet_ensemble_param_grid = {'k': [2, 4, 8, 16]}
    deep_transf.append(("tripletnet", tripletnet, tripletnet_param_grid))
    # deep_transf.append(("tripletnet_ensemble", tripletnet_ensemble, tripletnet_ensemble_param_grid))
    return deep_transf


def getMetrics(labels_names):
    """
    args:
        labels_names (dict): mapping from label code (int) to label name (str).
    """
    scoring = {'accuracy': 'accuracy',
               'f1_macro': 'f1_macro'}
    for code, name in labels_names.items():
        scoring['f-measure_%s' % name] = make_scorer(f1_score, average=None, labels=[code])

    return scoring


def combineTransformerClassifier(transformers, base_classifiers):
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=1.0/9, random_state=RANDOM_STATE)
    classifier = None

    for transf, base_classif in itertools.product(transformers, base_classifiers):
        transf_name, transf, transf_param_grid = transf
        base_classif_name, base_classif, base_classif_param_grid = base_classif
        classifier = Pipeline([('transformer', transf),
                               ('base_classifier', base_classif)],
                              memory=PIPELINE_CACHE_DIR)
        transf_param_grid = {"transformer__%s" % k: v for k, v in transf_param_grid.items()}
        base_classif_param_grid = {"base_classifier__%s" % k: v for k, v in base_classif_param_grid.items()}
        param_grid = {**transf_param_grid, **base_classif_param_grid}
        classifier = GridSearchCV_norefit(classifier, param_grid, scoring='f1_macro', cv=sampler)

        yield ('%s + %s' % (transf_name, base_classif_name), classifier)


def rpdbcs_cross_validate(estimator, X, y, groups, scoring=None, cv=None, verbose=0,
                          return_train_score=False, return_estimator=False):
    """
    Ensures that the validation dataset used by Gridsearch is a fold from the provided sampler.
    It substitutes the sampler of the Gridsearch (cv attribute).
    Only accepts KFold based cross validation.
    """
    from sklearn.metrics._scorer import _check_multimetric_scoring
    from sklearn.model_selection._validation import _fit_and_score, _aggregate_score_dicts
    from sklearn.base import clone

    if(not isinstance(cv, _BaseKFold)):
        return cross_validate(estimator, X, y, groups, scoring=scoring, cv=cv, verbose=verbose,
                              return_train_score=return_train_score, return_estimator=return_estimator)
    scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)
    folds = [(train_idxs, test_idxs) for train_idxs, test_idxs in cv.split(X, y, groups)]
    scores = []
    for i, (traindata, test_fold) in enumerate(folds):
        if(isinstance(estimator, BaseSearchCV)):
            validation_fold_indicator = np.where(np.isin(traindata, folds[i-1][1]), 0, -1)
            validation_sampler = PredefinedSplit(validation_fold_indicator)
            assert(validation_sampler.get_n_splits() == 1), validation_sampler.get_n_splits()
            estimator.cv = validation_sampler
        s = _fit_and_score(
            clone(estimator), X, y, scorers, traindata, test_fold, verbose, None,
            None, return_train_score=return_train_score,
            return_times=True, return_estimator=return_estimator,
            error_score=np.nan)
        scores.append(s)

    zipped_scores = list(zip(*scores))
    if return_train_score:
        train_scores = zipped_scores.pop(0)
        train_scores = _aggregate_score_dicts(train_scores)
    if return_estimator:
        fitted_estimators = zipped_scores.pop()
    test_scores, fit_times, score_times = zipped_scores
    test_scores = _aggregate_score_dicts(test_scores)

    ret = {'fit_time': np.array(fit_times),
           'score_time': np.array(score_times)}

    if return_estimator:
        ret['estimator'] = fitted_estimators

    for name in scorers:
        ret['test_%s' % name] = np.array(test_scores[name])
        if return_train_score:
            key = 'train_%s' % name
            ret[key] = np.array(train_scores[name])

    return ret


def main(save_file, D):
    global DEEP_CACHE_DIR, PIPELINE_CACHE_DIR

    X = np.expand_dims(D.asMatrix()[:, :6100], axis=1)
    Y, Ynames = D.getMulticlassTargets()
    group_ids = D.groupids('test')

    transformers = getDeepTransformers()
    base_classifiers = getBaseClassifiers(('normalizer', StandardScaler()))

    # sampler = StratifiedGroupKFold(5, shuffle=False)
    sampler = rpdbcsKFold(5, shuffle=True, random_state=RANDOM_STATE)
    # sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)

    scoring = getMetrics(Ynames)

    Results = {}
    for classifier_name, classifier in combineTransformerClassifier(transformers, base_classifiers):
        print(classifier_name)
        scores = rpdbcs_cross_validate(classifier, X, Y, groups=group_ids, scoring=scoring,
                                       cv=sampler)
        Results[classifier_name] = scores

    ictaifeats_names = getICTAI2016FeaturesNames()
    features = D.asDataFrame()[ictaifeats_names].values
    for classif_name, classifier, param_grid in base_classifiers:
        print(classif_name)
        # n_jobs: You may not want all your cores being used.
        classifier = GridSearchCV(classifier, param_grid, scoring='f1_macro', n_jobs=-1)
        scores = rpdbcs_cross_validate(classifier, features, Y, groups=group_ids,
                                       scoring=scoring, cv=sampler)
        # classifier.fit(features, Y)
        Results[classif_name] = scores

    results_asmatrix = []
    for classif_name, result in Results.items():
        print("===%s===" % classif_name)
        for rname, rs in result.items():
            if(rname.startswith('test_') or 'time' in rname):
                if(rname.startswith('test_')):
                    metric_name = rname.split('_', 1)[-1]
                else:
                    metric_name = rname
                print("%s: %f" % (metric_name, rs.mean()))
                for i, r in enumerate(rs):
                    results_asmatrix.append((classif_name, metric_name, i+1, r))

    if(save_file is not None):
        df = pd.DataFrame(results_asmatrix, columns=['classifier name', 'metric name', 'fold id', 'value'])
        df.to_csv(save_file, index=False)

        # for i, trained_model in enumerate(scores['estimator']):
        #     trained_model['encodding'].save_params("%s-%d.pt" % (save_file, i))
        # trained_model = trained_model['encodding']
        # with open("%s-%d.pkl" % (save_file, i), 'wb') as f:
        #     pickle.dump(trained_model, f)
    rmtree(PIPELINE_CACHE_DIR)
    rmtree(DEEP_CACHE_DIR)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdata', type=str, required=True)
    parser.add_argument('-o', '--outfile', type=str, required=False)
    args = parser.parse_args()

    D = loadRPDBCSData(args.inputdata)
    main(args.outfile, D)
