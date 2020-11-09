from sklearn.base import TransformerMixin, BaseEstimator
from joblib import Memory
import torch
from torch import nn
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset, getICTAI2016FeaturesNames
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, GroupShuffleSplit, StratifiedKFold
from rpdbcs.model_selection import StratifiedGroupKFold, rpdbcsKFold, GridSearchCV_norefit, rpdbcs_cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import skorch
from tripletnet.networks import TripletNetwork, TripletEnsembleNetwork
from tripletnet.datahandler import BalancedDataLoader
from torchvision import transforms
from tripletnet.networks import lmelloEmbeddingNet
import numpy as np
import pandas as pd
from tripletnet.callbacks import LoadEndState, LRMonitor, CleanNetCallback
import itertools
from tempfile import mkdtemp
from shutil import rmtree

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

DEEP_CACHE_DIR = mkdtemp()
PIPELINE_CACHE_DIR = mkdtemp()


def loadRPDBCSData(data_dir='data/data_classified_v6', nsigs=100000, normalize=True):
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=nsigs, npoints=10800, dtype=np.float32)
    D.discardMultilabel()
    targets, _ = D.getMulticlassTargets()
    # D.remove(np.where(targets == 3)[0])  # removes desalinhamento
    df = D.asDataFrame()
    # D.remove(df[(df['project name'] == 'Baker') & (df['bcs name'] == 'MA15')].index.values)
    print("Dataset length", len(D))
    if(normalize):
        D.normalize(37.28941975)
    D.shuffle()

    return D


def getBaseClassifiers(pre_pipeline=None):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    clfs = []
    knn = KNeighborsClassifier()
    knn_param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_param_grid = {'n_estimators': [100, 1000],
                     'max_features': [2, 3, 4, 5]}
    dtree = DecisionTreeClassifier(random_state=RANDOM_STATE, min_impurity_decrease=0.001)
    qda = QuadraticDiscriminantAnalysis()
    qda_param_grid = {'reg_param': [0.0, 1e-6, 1e-5]}
    clfs.append(("knn", knn, knn_param_grid))
    clfs.append(("DT", dtree, {}))
    # clfs.append(("RF", rf, rf_param_grid))
    clfs.append(("NB", GaussianNB(), {}))
    clfs.append(("QDA", qda, qda_param_grid))

    if(pre_pipeline is not None):
        return [(cname, Pipeline([pre_pipeline, ('base_clf', c)]), {"base_clf__%s" % k: v for k, v in pgrid.items()})
                for cname, c, pgrid in clfs]

    return clfs


def getCallbacks():
    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=DEEP_CACHE_DIR, monitor='train_loss_best')
    lrscheduler = skorch.callbacks.LRScheduler(policy=optim.lr_scheduler.StepLR,
                                               step_size=40, gamma=0.9, event_name=None)
    # É possível dar nomes ao callbacks para poder usar gridsearch neles: https://skorch.readthedocs.io/en/stable/user/callbacks.html#learning-rate-schedulers

    return [checkpoint_callback, LoadEndState(checkpoint_callback), lrscheduler, LRMonitor(), CleanNetCallback()]


def getDeepTransformers():
    def newEnsemble(n):
        return [TripletNetwork(lmelloEmbeddingNet, module__num_outputs=8, batch_size=80, init_random_state=i+100, **parameters)
                for i in range(n)]
    parameters = {
        'callbacks': getCallbacks(),
        'max_epochs': 80,
        'device': 'cuda',
        'optimizer': optim.Adam, 'optimizer__weight_decay': 1e-4, 'optimizer__lr': 1e-4,
        'train_split': None,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        # 'criterion': TripletNetwork.OnlineTripletLossWrapper,
        'margin_decay_value': 0.75}
    deep_transf = []
    tripletnet = TripletNetwork(lmelloEmbeddingNet, module__num_outputs=8, batch_size=80, **parameters)
    nets = newEnsemble(3)
    nets2 = newEnsemble(5)
    nets3 = newEnsemble(7)
    nets4 = newEnsemble(9)

    # tripletnet_param_grid = {'batch_size': [80],
    #                          'margin_decay_delay': [35, 50],
    #                          'module__num_outputs': [5, 8, 16, 32, 64, 128]}
    tripletnet_param_grid = {'batch_size': [80],
                             'margin_decay_delay': [40],
                             'module__num_outputs': [8],
                             'optimizer__lr': [1e-4, 5e-4, 1e-3]}

    # tripletnet_ensemble = TripletEnsembleNetwork(lmelloEmbeddingNet,
    #                                              optimizer=optim.Adam, optimizer__weight_decay=1e-4,
    #                                              device='cuda',
    #                                              batch_size=80, margin_decay_delay=50,
    #                                              train_split=None,
    #                                              iterator_train=BalancedDataLoader, iterator_train__num_workers=0, iterator_train__pin_memory=False,
    #                                              **parameters)
    # tripletnet_ensemble_param_grid = {'k': [2, 4, 8, 16],
    #                                   'module__num_outputs': [16, 32, 64]}
    # tripletnet_ensemble_param_grid = {'k': [4],
    #                                   'module__num_outputs': [16]}
    deep_transf.append(("ensemble_tripletnets_3", nets, tripletnet_param_grid))
    deep_transf.append(("ensemble_tripletnets_5", nets2, tripletnet_param_grid))
    # deep_transf.append(("ensemble_tripletnets_7", nets3, tripletnet_param_grid))
    # deep_transf.append(("ensemble_tripletnets_9", nets4, tripletnet_param_grid))
    deep_transf.append(("tripletnet", tripletnet, tripletnet_param_grid))
    # deep_transf.append(("tripletnet_ensemble", tripletnet_ensemble, tripletnet_ensemble_param_grid))
    return deep_transf


def getMetrics(labels_names):
    def foldcount(y1, y2):
        return len(y1)
    """
    args:
        labels_names (dict): mapping from label code (int) to label name (str).
    """
    scoring = {'accuracy': 'accuracy',
               'f1_macro': 'f1_macro',
               'precision_macro': 'precision_macro',
               'recall_macro': 'recall_macro',
               'log_loss': 'neg_log_loss',
               'fold count': make_scorer(foldcount)}
    for code, name in labels_names.items():
        scoring['f-measure_%s' % name] = make_scorer(f1_score, average=None, labels=[code])
        scoring['precision_%s' % name] = make_scorer(precision_score, average=None, labels=[code])
        scoring['recall_%s' % name] = make_scorer(recall_score, average=None, labels=[code])

    return scoring


def combineTransformerClassifier(transformers, base_classifiers):
    def buildPipeline(T, base_classif):
        return Pipeline([('transformer', T),
                         ('base_classifier', base_classif)],
                        #  ('base_classifier', GridSearchCV(base_classif, base_classif_param_grid))],
                        memory=PIPELINE_CACHE_DIR)

    def buildGridSearch(clf, transf_param_grid, base_classif_param_grid):
        transf_param_grid = {"transformer__%s" % k: v
                             for k, v in transf_param_grid.items()}
        base_classif_param_grid = {"base_classifier__%s" % k: v
                                   for k, v in base_classif_param_grid.items()}
        param_grid = {**transf_param_grid, **base_classif_param_grid}
        return GridSearchCV_norefit(clf, param_grid, scoring='f1_macro')

    for transf, base_classif in itertools.product(transformers, base_classifiers):
        transf_name, transf, transf_param_grid = transf
        base_classif_name, base_classif, base_classif_param_grid = base_classif
        if(isinstance(transf, list)):
            C = [("net%d" % i, buildPipeline(T, base_classif))
                 for i, T in enumerate(transf)]

            param_grid = {}
            for netname, _ in C:
                tpgrid = {"%s__transformer__%s" % (netname, k): v
                          for k, v in transf_param_grid.items()}
                bcgrid = {"%s__base_classifier__%s" % (netname, k): v
                          for k, v in base_classif_param_grid.items()}
                param_grid.update({**tpgrid, **bcgrid})
            eclf = VotingClassifier(estimators=C, voting='soft')
            classifier = GridSearchCV_norefit(eclf, param_grid, scoring='f1_macro')
        else:
            classifier = buildGridSearch(buildPipeline(transf, base_classif),
                                         transf_param_grid, base_classif_param_grid)

        yield ('%s + %s' % (transf_name, base_classif_name), classifier)


def main(save_file, D):
    global DEEP_CACHE_DIR, PIPELINE_CACHE_DIR
    import pandas as pd

    X = np.expand_dims(D.asMatrix()[:, :6100], axis=1)
    Y, Ynames = D.getMulticlassTargets()
    # Yset = enumerate(set(Y))
    # Y, Ymap = pd.factorize(Y)
    # Ynames = {i: Ynames[oldi] for i, oldi in enumerate(Ymap)}
    group_ids = D.groupids('bcs')

    transformers = getDeepTransformers()
    base_classifiers = getBaseClassifiers(('normalizer', StandardScaler()))

    # sampler = StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE)
    # sampler = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE)
    # sampler = rpdbcsKFold(5, shuffle=False)
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=RANDOM_STATE)
    # sampler = GroupShuffleSplit(n_splits=5, test_size=0.8, random_state=0)

    scoring = getMetrics(Ynames)

    Results = {}
    for classifier_name, classifier in combineTransformerClassifier(transformers, base_classifiers):
        print(classifier_name)
        Results[classifier_name] = rpdbcs_cross_validate(classifier, X, Y, groups=group_ids, scoring=scoring,
                                                         cv=sampler)
        # classifier.fit(X, Y)

    ictaifeats_names = getICTAI2016FeaturesNames()
    features = D.asDataFrame()[ictaifeats_names].values
    for classif_name, classifier, param_grid in base_classifiers:
        print(classif_name)
        # n_jobs: You may not want all your cores being used.
        classifier = GridSearchCV(classifier, param_grid, scoring='f1_macro', n_jobs=-1)
        scores = rpdbcs_cross_validate(classifier, features, Y, groups=group_ids,
                                       scoring=scoring, cv=sampler)
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
