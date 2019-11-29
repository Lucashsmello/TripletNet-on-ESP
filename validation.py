from rpdbcs.datahandler.dataset import readDataset, getICTAI2016FeaturesNames, getRPDBCS2FeaturesNames
from networks import lmelloEmbeddingNet, extract_embeddings
import torch
from sklearn.model_selection import cross_validate, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pandas
import numpy
from sklearn.pipeline import Pipeline
from feature_selector import ManualFeaturesSelector
from classifiers.augmented_classifier import AugmentedClassifier, ClassifierConvNet, RESET_FOLD_ID
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('once')


def getBaseClassifiers(use_normalization=True):
    clfs = []

    svm = sklearn.svm.SVC(kernel='rbf', random_state=0)
    svm = GridSearchCV(svm, {'C': [32, 128], 'gamma': [2, 8]}, cv=3)
    knn = KNeighborsClassifier()
    knn = GridSearchCV(knn, {'n_neighbors': [1, 3, 5, 7, 9]}, cv=3)

    clfs.append((svm, "SVM"))
    clfs.append((knn, "knn"))
    clfs.append((tree.DecisionTreeClassifier(random_state=0, min_impurity_decrease=0.0001), 'DT'))
    clfs.append((RandomForestClassifier(n_estimators=10,
                                        random_state=0, min_impurity_decrease=0.0001), "RF"))

    if(use_normalization):
        return [(Pipeline([
            ('scale', StandardScaler()),
            ('base_clf', c)]), cname) for c, cname in clfs]
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
    clfs = []
    #clfs.append((ClassifierConvNet(), "ConvNet"))
    baseclfs = getBaseClassifiers()
    for c, cname in baseclfs:
        clfs.append((AugmentedClassifier(c, num_predefined_feats,
                                         "saved_models/extractor_net"), "Aug-%s" % cname))
    return clfs


D = readDataset('data/data_classified_v4/freq.csv', 'data/data_classified_v4/labels.csv',
                remove_first=100, nsigs=10000, npoints=20000)
T, _ = D.getMulticlassTargets()
#D.remove((T[T == 0].index).values)
# D.normalize(f_hz="min")
D.shuffle()

rpdbcs2feats_names = getRPDBCS2FeaturesNames()
ictaifeats_names = getICTAI2016FeaturesNames()
#rpdbcs2feats = D.asDataFrame()[rpdbcs2feats_names].values
#ictaifeats = D.asDataFrame()[ictaifeats_names].values
allfeats_names = list(dict.fromkeys(rpdbcs2feats_names+ictaifeats_names))
allfeats = D.asDataFrame()[allfeats_names].values
Feats = numpy.concatenate((allfeats, D.asMatrix()), axis=1)


rpdbcs2feats_sel = [allfeats_names.index(fname) for fname in rpdbcs2feats_names]
ictaifeats_sel = [allfeats_names.index(fname) for fname in ictaifeats_names]
allfeats_sel = list(range(len(allfeats_names)))
frequencyfeats_sel = list(range(len(allfeats_names), len(allfeats_names) + 11028))
clfs = []
deep_clfs = getClassifiers(getDeepClassifiers(len(ictaifeats_sel)),
                           ictaifeats_sel+frequencyfeats_sel,
                           use_normalization=False)
clfs += deep_clfs
clfs += getClassifiers(getBaseClassifiers(), ictaifeats_sel)


targets, targets_name = D.getMulticlassTargets()
metrics = {"F-%d" % i: make_scorer(f1_score, labels=[i], average='macro') for i in range(5)}
metrics['f1_macro'] = 'f1_macro'
#metrics = ['precision_macro', 'recall_macro', 'f1_macro', 'f1_micro']
results_test = {m: [] for m in metrics}
results_train = {m: [] for m in metrics}
results_test_std = {m: [] for m in metrics}
sampler = StratifiedKFold(10, shuffle=True, random_state=222)
for clf, name in clfs:
    RESET_FOLD_ID()
    scores = cross_validate(clf, Feats, targets, scoring=metrics,
                            cv=sampler, return_train_score=True)
    for m in metrics:
        results_test[m].append(scores["test_%s" % m].mean())
        results_train[m].append(scores["train_%s" % m].mean())
        # results_std[m].append(scores["test_%s" % m].std())

df = pandas.DataFrame(results_test, index=list(zip(*clfs))[1])
# df_std = pandas.DataFrame(results_std, index=list(zip(*clfs))[1])
print(df)
df = pandas.DataFrame(results_train, index=list(zip(*clfs))[1])
print(df)
print(targets_name)
# print(df_std)
# num_outputs = 7
# triplet_model = lmelloEmbeddingNet(num_outputs)
# triplet_model.cuda()
