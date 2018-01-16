# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:40:12 2018

@author: 1008
"""

import numpy as np
import pandas as pd

SEED = 222
np.random.seed(SEED)

df = pd.read_csv('input.csv')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def get_train_test(test_size=0.95):
    y = 1 * (df.cand_pty_affiliation == "REP")
    X = df.drop(["cand_pty_affiliation"], axis=1)
    X = df.get_dummies(X, sparse=True)
    X.drop(X.columns[X.std() == 0], axis=1, inplace=True)
    return train_test_split(X, y, test_size=test_size, random_state=SEED)

xtrain, xtest, ytrain, ytest = get_train_test()

print("\nExample data:")
df.head()