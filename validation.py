from rpdbcs.datahandler.dataset import readDataset, getICTAI2016FeaturesNames, getRPDBCS2FeaturesNames
from networks import lmelloEmbeddingNet, extract_embeddings
import torch
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy
from sklearn.pipeline import Pipeline
from feature_selector import ManualFeaturesSelector


def getBaseClassifiers():
    clfs = []

    clfs.append((svm.SVC(kernel='linear', C=1, random_state=0), "SVM"))
    clfs.append((KNeighborsClassifier(n_neighbors=3), "3nn"))

    return clfs


def getClassifiers(base_clfs, selectedfeats):
    ret = []
    for clf in base_clfs:
        c = Pipeline([
            ('feature_selection', ManualFeaturesSelector(selectedfeats)),
            ('classification', clf[0])])
        ret.append((c, clf[1]))  # FIXME: make a new name base on the manual feature selection
    return ret


D = readDataset('data/data_classified_v4/freq.csv', 'data/data_classified_v4/labels.csv',
                remove_first=100, nsigs=10000, npoints=20000)
D.normalize(f_hz="min")
D.shuffle()

rpdbcs2feats_names = getRPDBCS2FeaturesNames()
ictaifeats_names = getICTAI2016FeaturesNames()
#rpdbcs2feats = D.asDataFrame()[rpdbcs2feats_names].values
#ictaifeats = D.asDataFrame()[ictaifeats_names].values
allfeats_names = list(dict.fromkeys(rpdbcs2feats_names+ictaifeats_names))
print(allfeats_names)
allfeats = D.asDataFrame()[allfeats_names].values
Feats = numpy.concatenate((allfeats, D.asMatrix()), axis=1)


#rpdbcs2feats_sel = [allfeats_names.index(fname) for fname in rpdbcs2feats_names]
allfeats_sel = list(range(len(allfeats_names)))
clfs = getClassifiers(getBaseClassifiers(), allfeats_sel)

targets, targets_name = D.getMulticlassTargets()
metrics = ['precision_macro', 'recall_macro', 'f1_macro']
results = {m: [] for m in metrics}
for clf, name in clfs:
    scores = cross_validate(clf, Feats, targets, scoring=metrics, cv=5, return_train_score=False)
    for m in metrics:
        results[m].append(scores["test_%s" % m].mean())

df = pandas.DataFrame(results, index=list(zip(*clfs))[1])
print(df)
# num_outputs = 7
# triplet_model = lmelloEmbeddingNet(num_outputs)
# triplet_model.cuda()
