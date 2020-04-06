from rpdbcs.datahandler.dataset import readDataset, getICTAI2016FeaturesNames, getRPDBCS2FeaturesNames
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV, KFold, StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.decomposition import PCA
from pandas import DataFrame
import numpy
from sklearn.pipeline import Pipeline
from tripletnet.feature_selector import ManualFeaturesSelector
from tripletnet.classifiers.augmented_classifier import AugmentedClassifier, ClassifierConvNet, ClassifierConvNet2, RESET_FOLD_ID
from sklearn.metrics import make_scorer
import time
import warnings
warnings.simplefilter(action='once', category=FutureWarning)
warnings.simplefilter(action='once', category=DeprecationWarning)
warnings.filterwarnings('once')

RANDOM_STATE = 0


class testclf:
    def __init__(self, myparam=0):
        self.myparam = myparam

    def fit(self, X, y):
        print(">>>trained with %d samples and myparam=%d." % (len(y), self.myparam))
        return self

    def get_params(self, deep=True):
        return {'myparam': self.myparam}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if(parameter == 'myparam'):
                self.myparam = value
        return self

    def predict(self, X):
        return numpy.random.randint(0, 4, size=len(X))


def getBaseClassifiers(pre_pipeline=None):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    clfs = []
    scorer = 'f1_macro'
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=1.0/9, random_state=RANDOM_STATE)
    # sampler = StratifiedKFold(9, shuffle=False)

    svm = sklearn.svm.SVC(kernel='rbf', random_state=RANDOM_STATE)
    svm = GridSearchCV(svm,
                       {'C': [2**5, 2**7, 2**13, 2**15],
                        'gamma': [2, 8]},
                       scoring=scorer, cv=sampler, n_jobs=6)
    knn = KNeighborsClassifier()
    knn = GridSearchCV(knn,
                       {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]},
                       scoring=scorer, cv=sampler, n_jobs=6)
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf = GridSearchCV(rf, {'n_estimators': [100, 1000],
                           'max_features': [1, 2, 3, 4, 5]},
                      #    'min_impurity_decrease': [1e-4, 1e-3]},
                      scoring=scorer, cv=sampler, n_jobs=6)
    dtree = tree.DecisionTreeClassifier(random_state=RANDOM_STATE)
    dtree = GridSearchCV(dtree,
                         {
                             'max_leaf_nodes': [10, 100, 1000],
                             'max_depth': [3, 6, 9, 12, 15]
                         }
                         )
    qda = QuadraticDiscriminantAnalysis()
    qda = GridSearchCV(qda,
                       {
                           'reg_param': [0.0, 1e-6, 1e-5]
                       },
                       scoring=scorer, cv=sampler, n_jobs=6)

    clfs.append((svm, "SVM"))
    clfs.append((knn, "knn"))
    clfs.append((dtree, "DT"))
    clfs.append((rf, "RF"))
    clfs.append((GaussianNB(), "NB"))
    clfs.append((qda, "QDA"))
    # ttt = testclf()
    # ttt = GridSearchCV(ttt, {'myparam': [1, 2, 3]}, scoring=scorer, cv=sampler)
    # clfs.append((ttt, "testclf"))

    if(pre_pipeline is not None):
        return [(Pipeline([pre_pipeline,
                           ('base_clf', c)]), cname)
                for c, cname in clfs]

    return clfs


def getClassifiers(base_clfs, selectedfeats, use_normalization=False):
    ret = []
    for clf in base_clfs:
        if(use_normalization):
            c = Pipeline([
                ('feature_selection', ManualFeaturesSelector(selectedfeats)),
                ('scale', StandardScaler()),
                ('classification', clf[0])])
        else:
            c = Pipeline([
                ('feature_selection', ManualFeaturesSelector(selectedfeats)),
                ('classification', clf[0])])
        ret.append((c, clf[1]))  # FIXME: make a new name base on the manual feature selection
    return ret


def getDeepClassifiers(num_predefined_feats):
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=1.0/9, random_state=RANDOM_STATE)
    scorer = 'f1_macro'
    clfs = []
    # clfs.append((ClassifierConvNet(base_dir="saved_models/ClassifierConvNet", nclasses=5), "ConvNet"))
    baseclfs_std = getBaseClassifiers(('scale', StandardScaler()))
    baseclfs = getBaseClassifiers()
    for c, cname in baseclfs:
        augclf = AugmentedClassifier(c, num_predefined_feats, "saved_models/extractor_net")
        augclf = GridSearchCV(augclf, {
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'num_subepochs': [20, 30, 40],
            'batch_size': [4, 8, 16]
        }, scoring=scorer, cv=sampler)
        clfs.append((augclf, "Aug-%s" % cname))

    # for c, cname in baseclfs_std:
    #     augclf2 = ClassifierConvNet2(
    #         base_dir="saved_models/ClassifierConvNet", nclasses=5, base_classif=c)
    #     augclf2 = GridSearchCV(augclf2, {
    #         'learning_rate': [1e-4, 1e-3, 1e-2],
    #         'num_steps_decay': [20, 30, 40],
    #         'batch_size': [128, 256, 512]
    #     }, scoring=scorer, cv=sampler)
    #     clfs.append((augclf2, "ConvNet-%s" % cname))
    return clfs


D = readDataset('data/data_classified_v6/freq.csv', 'data/data_classified_v6/labels.csv',
                remove_first=100, nsigs=20000, npoints=10800)
targets, _ = D.getMulticlassTargets()
D.remove((targets[targets == 0].index).values)
D.normalize(f_hz="min")
# D.shuffle()
targets, targets_name = D.getMulticlassTargets()
print(targets_name)


rpdbcs2feats_names = getRPDBCS2FeaturesNames()
ictaifeats_names = getICTAI2016FeaturesNames()
# rpdbcs2feats = D.asDataFrame()[rpdbcs2feats_names].values
# ictaifeats = D.asDataFrame()[ictaifeats_names].values
allfeats_names = list(dict.fromkeys(rpdbcs2feats_names+ictaifeats_names))
allfeats = D.asDataFrame()[allfeats_names].values
Feats = numpy.concatenate((allfeats, D.asMatrix()), axis=1)


rpdbcs2feats_sel = [allfeats_names.index(fname) for fname in rpdbcs2feats_names]
ictaifeats_sel = [allfeats_names.index(fname) for fname in ictaifeats_names]
# print(len(ictaifeats_sel))
allfeats_sel = list(range(len(allfeats_names)))
frequencyfeats_sel = list(range(len(allfeats_names), len(allfeats_names) + 6100))  # 5076
clfs = []

# deep_clfs_feats = ictaifeats_sel+frequencyfeats_sel
deep_clfs_feats = frequencyfeats_sel
deep_clfs = getClassifiers(getDeepClassifiers(len(deep_clfs_feats)-len(frequencyfeats_sel)),
                           deep_clfs_feats,
                           use_normalization=False)
clfs += deep_clfs
clfs += getClassifiers(getBaseClassifiers(), ictaifeats_sel, use_normalization=True)


# metrics = {"F-%d" % i: make_scorer(f1_score, labels=[i], average='macro') for i in range(5)}
"""
Valid Options:
    ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'brier_score_loss', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'max_error',
        'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']
"""
metrics = ['precision_macro', 'recall_macro', 'f1_macro',
           'f1_micro', 'accuracy']
metrics = {m: m for m in metrics}
results_test = {m: [] for m in metrics}
results_test_folds = {"f1_macro": [], "clf_name": []}
results_train = {m: [] for m in metrics}
results_test_std = {m: [] for m in metrics}
# y_diff = {}

sampler = StratifiedKFold(10, shuffle=False)
for clf, name in clfs:
    t1 = time.time()
    RESET_FOLD_ID()
    scores = cross_validate(clf, Feats, targets, scoring=metrics,
                            cv=sampler, return_train_score=True, return_estimator=False)
    t2 = time.time()
    print("----------%s-----------(%.2fseconds)" % (name, t2-t1))

    for m in metrics:
        results_test[m].append(scores["test_%s" % m].mean())
        results_train[m].append(scores["train_%s" % m].mean())
        results_test_std[m].append(scores["test_%s" % m].std())
    for sc in scores["test_f1_macro"]:
        results_test_folds["f1_macro"].append(sc)
        results_test_folds["clf_name"].append(name)
    '''
    y_pred = cross_val_predict(clf, Feats, targets, cv=sampler)
    print('>>>', len(y_pred))
    for m_name, metric in metrics.items():
        if('macro' in m_name):
            results_test[m_name].append(metric(targets, y_pred, average='macro'))
        elif('micro' in m_name):
            results_test[m_name].append(metric(targets, y_pred, average='micro'))
        else:
            results_test[m_name].append(metric(targets, y_pred))
    y_diff[name] = y_pred == targets
    '''
print("Cross validation finished")

df_test = DataFrame(results_test, index=list(zip(*clfs))[1])
df_std = DataFrame(results_test_std, index=list(zip(*clfs))[1])
df_train = DataFrame(results_train, index=list(zip(*clfs))[1])
df_test_folds = DataFrame(results_test_folds)
print(df_test)
# print(df_std)
print(df_train)
df_test.to_csv('results/performance_test.csv')
df_train.to_csv('results/performance_train.csv')
df_std.to_csv('results/performance_test_std.csv')
df_test_folds.to_csv('results/performance_test_folds.csv')

'''
df_preds = DataFrame(y_diff)
df_preds['Signal id'] = D.asDataFrame()['Signal id']
df_preds.to_csv('results/predictions.csv', sep=';')
'''
