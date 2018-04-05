# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 18:43:34 2018

@author: guilherme.santos

Obs: Para a base de dados kc2, eu pre-processei pelo próprio excel as classes: de 'yes' para 1 e 'no' para 0
"""
#test import
import os
#test import

import pandas as pd
import math
import operator
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

datatrieve_data = pd.read_csv("dataset/q1/datatrieve.csv",header=None)
kc2_data = pd.read_csv("dataset/q1/kc2.csv",header=None)

#datatrieve dataset X and y
X_datatrieve = preprocessing.scale(datatrieve_data.iloc[:,:-1].values)
y_datatrieve = datatrieve_data.iloc[:,8].values

#kc2 dataset X and y
X_kc2 = preprocessing.scale(kc2_data.iloc[:,:-1].values)
y_kc2 = kc2_data.iloc[:,21].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X_datatrieve,y_datatrieve, test_size = 0.2, random_state = 0)

neigh = KNeighborsClassifier( n_neighbors=3)
neigh.fit(X_train, y_train)

print(neigh.kneighbors([X_test[3]],n_neighbors=5))
#skf = StratifiedKFold(n_splits=5)

def euclidianDistance(v1, v2):
    return math.sqrt(sum(pow((v1 - v2),2)))
