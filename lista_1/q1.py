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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

datatrieve_data = pd.read_csv("dataset/datatrieve.csv",header=None)
kc2_data = pd.read_csv("dataset/kc2.csv",header=None)

#datatrieve dataset X and y
X_datatrieve = preprocessing.scale(datatrieve_data.iloc[:,:-1].values)
y_datatrieve = datatrieve_data.iloc[:,8].values

#kc2 dataset X and y
X_kc2 = preprocessing.scale(kc2_data.iloc[:,:-1].values)
y_kc2 = kc2_data.iloc[:,21].values


skf = StratifiedKFold(n_splits=5)

#for train_indexes, test_indexes in skf.split(X_datatrieve,y_datatrieve):
#     print("TRAIN:", X_datatrieve[train_indexes], "TEST:", y_datatrieve[test_indexes], "\n\n\n -----------------")
#     os.system("pause")

for train_indexes, test_indexes in skf.split(X_kc2,y_kc2):
     print("TRAIN:", X_kc2[train_indexes], "TEST:", y_kc2[test_indexes], "\n\n\n -----------------")
     os.system("pause")
     
#for train, test in kf.split(dataset_datatrieve):
#    print("%s %s" % (train, test))
#kf.split(dataset_datatrieve)