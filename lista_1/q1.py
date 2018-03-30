# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 18:43:34 2018

@author: guilherme.santos
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

dataset_datatrieve = pd.read_csv("dataset/datatrieve.csv")
dataset_kc2 = pd.read_csv("dataset/kc2.csv")

kf = KFold(n_splits=5,shuffle=True, random_state=32)

for train, test in kf.split(dataset_datatrieve):
    print("%s %s" % (train, test))
#kf.split(dataset_datatrieve)