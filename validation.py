from sklearn.base import TransformerMixin, BaseEstimator
import torch
from torch import nn
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset, getICTAI2016FeaturesNames
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, GroupShuffleSplit, StratifiedKFold, cross_validate
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
from tripletnet.classifiers.TorchBaggingClassifier import TorchBaggingClassifier
from adabelief_pytorch import AdaBelief
import os

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
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=1000, n_jobs=-1)
    rf_param_grid = {'max_features': [2, 3, 4, 5]}
    dtree = DecisionTreeClassifier(random_state=RANDOM_STATE, min_impurity_decrease=0.001)
    qda = QuadraticDiscriminantAnalysis()
    qda_param_grid = {'reg_param': [0.0, 1e-6, 1e-5]}
    clfs.append(("knn", knn, knn_param_grid))
    #clfs.append(("DT", dtree, {}))
    clfs.append(("RF", rf, rf_param_grid))
    #clfs.append(("NB", GaussianNB(), {}))
    clfs.append(("QDA", qda, qda_param_grid))

    if(pre_pipeline is not None):
        return [(cname, Pipeline([pre_pipeline, ('base_clf', c)]), {"base_clf__%s" % k: v for k, v in pgrid.items()})
                for cname, c, pgrid in clfs]

    return clfs


def getCallbacks():
    checkpoint_callback = skorch.callbacks.Checkpoint(
        dirname=DEEP_CACHE_DIR, monitor='non_zero_triplets_best')  # monitor='train_loss_best')
    lrscheduler = skorch.callbacks.LRScheduler(policy=optim.lr_scheduler.StepLR,
                                               step_size=25, gamma=0.8, event_name=None)
    # É possível dar nomes ao callbacks para poder usar gridsearch neles: https://skorch.readthedocs.io/en/stable/user/callbacks.html#learning-rate-schedulers

    callbacks = [('non_zero_triplets', skorch.callbacks.PassthroughScoring(name='non_zero_triplets', on_train=True))]
    callbacks += [checkpoint_callback, LoadEndState(checkpoint_callback), lrscheduler, LRMonitor(), CleanNetCallback()]

    return callbacks


def getDeepTransformers():
    global DEEP_CACHE_DIR

    def newEnsemble(n):
        return [TripletNetwork(module__num_outputs=8, init_random_state=i+100, **parameters)
                for i in range(n)]

    optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                            'eps': 1e-16, 'betas': (0.9, 0.999),
                            'weight_decouple': True, 'rectify': False}
    optimizer_parameters = {"optimizer__"+key: v for key, v in optimizer_parameters.items()}
    optimizer_parameters['optimizer'] = AdaBelief

    # optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3}
    # optimizer_parameters = {"optimizer__"+key: v for key, v in optimizer_parameters.items()}
    # optimizer_parameters['optimizer'] = optim.Adam

    parameters = {
        'callbacks': getCallbacks(),
        'device': 'cuda',
        'module': lmelloEmbeddingNet,
        'max_epochs': 300,
        'train_split': None,
        'batch_size': 80,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        # 'criterion__triplet_selector': siamese_triplet.utils.HardestNegativeTripletSelector(1.0),
        'margin_decay_value': 0.75, 'margin_decay_delay': 100}
    parameters = {**parameters, **optimizer_parameters}
    deep_transf = []
    tripletnet = TripletNetwork(module__num_outputs=8, init_random_state=100, **parameters)

    # tripletnet_param_grid = {'batch_size': [80],
    #                          'margin_decay_delay': [35, 50],
    #                          'module__num_outputs': [5, 8, 16, 32, 64, 128]}
    tripletnet_param_grid = {'batch_size': [80],
                             'margin_decay_delay': [50],
                             'module__num_outputs': [8],
                             'optimizer__lr': [1e-4, 5e-4, 1e-3]}

    # tripletnet_ensemble_param_grid = {'k': [2, 4, 8, 16],
    #                                   'module__num_outputs': [16, 32, 64]}
    # tripletnet_ensemble_param_grid = {'k': [4],
    #                                   'module__num_outputs': [16]}
    # ensemble_name = "ensemble_voting"
    ensemble_name = "ensemble_bagging"
    for i in range(21, 3-1, -2):
        #nets = newEnsemble(i)
        #deep_transf.append(("%s_tripletnets_%d" % (ensemble_name, i), nets, tripletnet_param_grid))
        tripletnet_dropouton = [TripletNetwork(
            module__num_outputs=8, init_random_state=100, dropout_on=True, **parameters)] * i
        deep_transf.append(("tripletnet_dropouton_%d" % i, tripletnet_dropouton, tripletnet_param_grid))
    #deep_transf.append(("tripletnet", tripletnet, tripletnet_param_grid))
    return deep_transf


def createNeuralClassifier():
    """
    Common neural net classifier.
    """
    from siamese_triplet.networks import ClassificationNet

    class MyNeuralNetClassifier(skorch.NeuralNetClassifier):
        def __init__(self, module, init_random_state, cache_dir,
                     *args,
                     criterion=torch.nn.NLLLoss,
                     train_split=None,
                     classes=None,
                     optimizer__lr=1e-3,
                     **kwargs):
            super().__init__(module, *args, criterion=criterion, train_split=train_split,
                             classes=classes, optimizer__lr=optimizer__lr, **kwargs)
            self.init_random_state = init_random_state
            self.cache_dir = cache_dir

        def initialize(self):
            if(self.init_random_state is not None):
                np.random.seed(self.init_random_state)
                torch.cuda.manual_seed(self.init_random_state)
                torch.manual_seed(self.init_random_state)
            return super().initialize()

        def get_cache_filename(self):
            return "%s/%d-%.5f-%s.pkl" % (self.cache_dir, self.optimizer__lr, self.init_random_state, self.module.__name__[:8])

        def fit(self, X, y, **fit_params):
            cache_filename = self.get_cache_filename()
            if(os.path.isfile(cache_filename)):
                if not self.warm_start or not self.initialized_:
                    self.initialize()
                self.load_params(cache_filename)
                return self
            super().fit(X, y, **fit_params)
            self.save_params(cache_filename)
            return self

    def newEnsemble(n):
        estimators = [GridSearchCV_norefit(MyNeuralNetClassifier(ClassificationNet, init_random_state=100+i, cache_dir=DEEP_CACHE_DIR, **parameters),
                                           param_grid=grid_params, scoring='f1_macro', cv=gridsearch_sampler)
                      for i in range(n)]
        # estimators = [MyNeuralNetClassifier(ClassificationNet, init_random_state=100+i, cache_dir=DEEP_CACHE_DIR, **parameters)
        #              for i in range(n)]
        estimators = [('net%d' % i, clf) for i, clf in enumerate(estimators)]
        return VotingClassifier(estimators=estimators, voting='soft')

    gridsearch_sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.11, random_state=RANDOM_STATE)
    grid_params = {'optimizer__lr': [1e-4, 5e-4, 1e-3]}

    optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                            'eps': 1e-16, 'betas': (0.9, 0.999),
                            'weight_decouple': True, 'rectify': False}
    optimizer_parameters = {"optimizer__"+key: v for key, v in optimizer_parameters.items()}
    optimizer_parameters['optimizer'] = AdaBelief

    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=DEEP_CACHE_DIR, monitor='train_loss_best')
    # É possível dar nomes ao callbacks para poder usar gridsearch neles: https://skorch.readthedocs.io/en/stable/user/callbacks.html#learning-rate-schedulers

    callbacks = [checkpoint_callback, LoadEndState(checkpoint_callback)]

    parameters = {
        'callbacks': callbacks,
        'device': 'cuda',
        'max_epochs': 300,
        'train_split': None,
        'batch_size': 80,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        'module__embedding_net': lmelloEmbeddingNet(8), 'module__n_classes': 5}
    parameters = {**parameters, **optimizer_parameters}
    convnet = skorch.NeuralNetClassifier(ClassificationNet, **parameters)

    ret = []
    ret.append(('convnet', GridSearchCV_norefit(convnet, param_grid=grid_params, scoring='f1_macro', cv=gridsearch_sampler)))
    for i in range(13, 2, -2):
        ret.append(('ensemble_convnet_%d' % i, newEnsemble(i)))

    return ret


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
    gridsearch_sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.11, random_state=RANDOM_STATE)

    def buildPipeline(T, base_classif, base_classif_param_grid=None):
        if(base_classif_param_grid is not None):
            base_classif = GridSearchCV(base_classif, base_classif_param_grid, cv=gridsearch_sampler, n_jobs=-1)

        return Pipeline([('transformer', T),
                         ('base_classifier', base_classif)],
                        memory=PIPELINE_CACHE_DIR)

    def buildGridSearch(clf, transf_param_grid, base_classif_param_grid):
        transf_param_grid = {"transformer__%s" % k: v
                             for k, v in transf_param_grid.items()}
        base_classif_param_grid = {"base_classifier__%s" % k: v
                                   for k, v in base_classif_param_grid.items()}
        param_grid = {**transf_param_grid, **base_classif_param_grid}
        return GridSearchCV_norefit(clf, param_grid, scoring='f1_macro', cv=gridsearch_sampler)

    rets = []
    for transf, base_classif in itertools.product(transformers, base_classifiers):
        transf_name, transf, transf_param_grid = transf
        base_classif_name, base_classif, base_classif_param_grid = base_classif
        if(isinstance(transf, list)):
            C = [("net%d" % i, buildPipeline(T, base_classif, base_classif_param_grid))
                 for i, T in enumerate(transf)]

            param_grid = {}
            for netname, _ in C:
                tpgrid = {"%s__transformer__%s" % (netname, k): v
                          for k, v in transf_param_grid.items()}
                bcgrid = {"%s__base_classifier__%s" % (netname, k): v
                          for k, v in base_classif_param_grid.items()}
                param_grid.update({**tpgrid, **bcgrid})
            if('voting' in transf_name or 'dropouton' in transf_name):
                classifier = VotingClassifier(estimators=C, voting='soft')
            elif('bagging' in transf_name):
                classifier = TorchBaggingClassifier(base_estimator=C[0][1], n_estimators=len(C), bootstrap=True,
                                                    bootstrap_features=False, random_state=RANDOM_STATE)
            else:
                raise Exception('ensemble "%s" not recognized!' % transf_name)
            # classifier = GridSearchCV_norefit(eclf, param_grid, scoring='f1_macro')
        else:
            # classifier = buildGridSearch(buildPipeline(transf, base_classif),
            #                             transf_param_grid, base_classif_param_grid)
            classifier = buildPipeline(transf, base_classif, base_classif_param_grid)

        final_name = '%s + %s' % (transf_name, base_classif_name)
        yield (final_name, classifier)


def main(save_file, D):
    global DEEP_CACHE_DIR, PIPELINE_CACHE_DIR, ENSEMBLE_CACHE_DIR
    import pandas as pd

    TEST_TRIPLETNET = True
    TEST_CONVNET = False
    TEST_BASECLASSIFIERS = True

    X = np.expand_dims(D.asMatrix()[:, :6100], axis=1)
    Y, Ynames = D.getMulticlassTargets()
    # Yset = enumerate(set(Y))
    # Y, Ymap = pd.factorize(Y)
    # Ynames = {i: Ynames[oldi] for i, oldi in enumerate(Ymap)}
    # group_ids = D.groupids('bcs')

    transformers = getDeepTransformers()
    base_classifiers = getBaseClassifiers(('normalizer', StandardScaler()))

    sampler = StratifiedKFold(10, shuffle=True, random_state=RANDOM_STATE)
    # sampler = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE)
    # sampler = rpdbcsKFold(5, shuffle=False)
    # sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    # sampler = GroupShuffleSplit(n_splits=5, test_size=0.8, random_state=0)
    gridsearch_sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.11, random_state=RANDOM_STATE)

    scoring = getMetrics(Ynames)

    Results = {}

    if(TEST_CONVNET):
        classifiers = createNeuralClassifier()
        for clf_name, clf in classifiers:
            print(clf_name, clf.__class__.__name__)
            Results[clf_name] = cross_validate(clf, X, Y, scoring=scoring,
                                               cv=sampler)

    if(TEST_TRIPLETNET):
        for classifier_name, classifier in combineTransformerClassifier(transformers, base_classifiers):
            print(classifier_name)
            Results[classifier_name] = cross_validate(classifier, X, Y, scoring=scoring,
                                                      cv=sampler)

    if(TEST_BASECLASSIFIERS):
        ictaifeats_names = getICTAI2016FeaturesNames()
        features = D.asDataFrame()[ictaifeats_names].values
        for classif_name, classifier, param_grid in base_classifiers:
            print(classif_name)
            # n_jobs: You may not want all your cores being used.
            classifier = GridSearchCV(classifier, param_grid, scoring='f1_macro', n_jobs=-1, cv=gridsearch_sampler)
            scores = cross_validate(classifier, features, Y, scoring=scoring, cv=sampler)
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
