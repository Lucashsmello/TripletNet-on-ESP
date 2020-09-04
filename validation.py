import torch
from torch import nn
import torch.optim as optim
from rpdbcs.datahandler.dataset import readDataset, getICTAI2016FeaturesNames
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_validate, StratifiedShuffleSplit, ShuffleSplit, GroupKFold, GridSearchCV
from sklearn.model_selection import cross_validate
from rpdbcs.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

RANDOM_STATE = 2
np.random.seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def loadRPDBCSData(data_dir='data/data_classified_v6', nsigs=100000):
    D = readDataset('%s/freq.csv' % data_dir, '%s/labels.csv' % data_dir,
                    remove_first=100, nsigs=nsigs, npoints=10800, dtype=np.float32)
    targets, _ = D.getMulticlassTargets()
    D.remove(np.where(targets == 3)[0])
    print("Dataset length", len(D))
    # D.normalize(37.28941975)
    # D.shuffle()

    return D


def getBaseClassifiers(pre_pipeline=None):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    clfs = []
    scorer = 'f1_macro'
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=1.0/9, random_state=RANDOM_STATE)
    # sampler = StratifiedKFold(9, shuffle=False)

    # svm = sklearn.svm.SVC(kernel='rbf', random_state=RANDOM_STATE)
    # svm = GridSearchCV(svm,
    #                    {'C': [2**5, 2**7, 2**13, 2**15],
    #                     'gamma': [2, 8]},
    #                    scoring=scorer, cv=sampler, n_jobs=6)
    knn = KNeighborsClassifier()
    knn = GridSearchCV(knn,
                       {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]},
                       scoring=scorer, cv=sampler, n_jobs=6)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf = GridSearchCV(rf, {'n_estimators': [100, 200],
                           'max_features': [1, 2, 3, 4, 5]},
                      #    'min_impurity_decrease': [1e-4, 1e-3]},
                      scoring=scorer, cv=sampler, n_jobs=6)
    dtree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dtree = GridSearchCV(dtree,
                         {
                             'max_leaf_nodes': [10, 100],
                             'max_depth': [3, 6, 9, 12, 15]
                         }
                         )
    qda = QuadraticDiscriminantAnalysis()
    qda = GridSearchCV(qda,
                       {
                           'reg_param': [0.0, 1e-6, 1e-5]
                       },
                       scoring=scorer, cv=sampler, n_jobs=6)

    # clfs.append((svm, "SVM"))
    clfs.append(("knn", knn))
    clfs.append(("DT", dtree))
    clfs.append(("RF", rf))
    clfs.append(("NB", GaussianNB()))
    clfs.append(("QDA", qda))
    # ttt = testclf()
    # ttt = GridSearchCV(ttt, {'myparam': [1, 2, 3]}, scoring=scorer, cv=sampler)
    # clfs.append((ttt, "testclf"))

    if(pre_pipeline is not None):
        return [(Pipeline([pre_pipeline,
                           ('base_clf', c)]), cname)
                for c, cname in clfs]

    return clfs


DEEP_CACHE_DIR = mkdtemp()


def getDeepTransformers():
    global DEEP_CACHE_DIR

    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=DEEP_CACHE_DIR, monitor='train_loss_best')
    parameters = {
        'callbacks': [checkpoint_callback, LoadEndState(checkpoint_callback)],
        'max_epochs': 100,
        'batch_size': 125,
        'margin_decay_delay': 50,
        'margin_decay_value': 0.5}
    deep_transf = []
    tripletnet = TripletNetwork(lmelloEmbeddingNet,
                                optimizer=optim.Adam, optimizer__lr=1e-4, optimizer__weight_decay=1e-4,
                                module__num_outputs=8, device='cuda',
                                train_split=None,
                                criterion=TripletNetwork.OnlineTripletLossWrapper,
                                iterator_train=BalancedDataLoader, iterator_train__num_workers=3, iterator_train__pin_memory=True,
                                **parameters
                                )
    # tripletnet = TripletEnsembleNetwork(lmelloEmbeddingNet, k=4,
    #                                     optimizer=optim.Adam, optimizer__lr=1e-4, optimizer__weight_decay=1e-4,
    #                                     module__num_outputs=32, device='cuda',
    #                                     train_split=None,
    #                                     criterion=TripletNetwork.OnlineTripletLossWrapper,
    #                                     iterator_train=BalancedDataLoader, iterator_train__num_workers=3, iterator_train__pin_memory=True,
    #                                     **parameters)
    deep_transf.append(("tripletnet", tripletnet))
    return deep_transf


def main(save_file, D, method="orig"):
    global DEEP_CACHE_DIR

    X = np.expand_dims(D.asMatrix()[:, :6100], axis=1)
    Y, _ = D.getMulticlassTargets()
    Y, Ynames = pd.factorize(Y)
    group_ids = D.groupids('test')

    transformers = getDeepTransformers()
    base_classifiers = getBaseClassifiers()

    sksampler = StratifiedGroupKFold(5, shuffle=False)
    # sksampler = StratifiedKFold(10, shuffle=False)
    # sksampler = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    cachedir = mkdtemp()

    scoring = ['accuracy', 'f1_macro']
    Results = {}
    for transf, base_classif in itertools.product(transformers, base_classifiers):
        transf_name, transf = transf
        classif_name, base_classif = base_classif
        classifier = Pipeline([('encodding', transf),
                               ('classifier', base_classif)],
                              memory=cachedir)
        scores = cross_validate(classifier, X, Y, groups=group_ids, scoring=scoring,
                                cv=sksampler, return_estimator=False)
        Results['%s + %s' % (transf_name, classif_name)] = scores

    ictaifeats_names = getICTAI2016FeaturesNames()
    features = D.asDataFrame()[ictaifeats_names].values
    for classif_name, base_classif in base_classifiers:
        scores = cross_validate(base_classif, features, Y, groups=group_ids,
                                scoring=scoring, cv=sksampler)
        Results[classif_name] = scores
    # classifier = GridSearchCV(classifier,
    #                           {'encodding__optimizer__lr': [1e-4]}, scoring='f1_macro',
    #                           cv=StratifiedShuffleSplit(n_splits=1, test_size=1.0/9, random_state=RANDOM_STATE))

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
    rmtree(cachedir)
    rmtree(DEEP_CACHE_DIR)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdata', type=str, required=True)
    # parser.add_argument('--model', type=str, required=False, help='pre-trained model in pkl')
    parser.add_argument('-o', '--outfile', type=str, required=False)
    parser.add_argument('--method', type=str, choices=['orig', 'divconquer'], default='orig')
    args = parser.parse_args()

    D = loadRPDBCSData(args.inputdata)
    main(args.outfile, D, method=args.method)
    # main2(args.outfile, D, trained_model=args.model)
