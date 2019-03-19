from pyopf import OPFClassifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import sklearn
from scipy import spatial

import numpy as np
import gc

def my_distance(x,y):

    try:
        return np.sqrt(np.sum(np.power(x-y,2)))
    except RuntimeWarning as e:
        print(x-y)

def my_cos(x,y):
    d = spatial.distance.cosine(x, y+1e-9)
    return d


l =load_iris()

opf = OPFClassifier("cosine")
# opf = OPF('cos')
x = np.array(l['data'])
y = l['target']
ss = StratifiedKFold(2)
splits = list(ss.split(x, y))

# for i in range(100):
#
#     acc = []
#     for train_idx,test_idx in splits:
#         train_data = x[train_idx]
#         train_label = y[train_idx]
#
#         test_data = x[test_idx]
#         test_label = y[test_idx]
#
#         opf.fit(train_data, train_label)
#         preds = opf.predict(test_data)
#
#         acc.append(accuracy_score(test_label, preds))
#
#         prev_data = train_data.copy()
#
#     print("acc:", np.mean(acc))

for train_idx, test_idx in splits:
    train_data = x[train_idx]
    train_label = y[train_idx]

    test_data = x[test_idx]
    test_label = y[test_idx]

for i in range(10):

    opf.fit(train_data, train_label)
    preds = opf.predict(test_data)

    acc = accuracy_score(test_label, preds)

    prev_data = train_data.copy()

    print("acc:", acc)