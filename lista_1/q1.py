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


datatrieve_data = pd.read_csv("dataset/q1/datatrieve.csv",header=None)
kc2_data = pd.read_csv("dataset/q1/kc2.csv",header=None)

#datatrieve dataset X and y
X_datatrieve = preprocessing.scale(datatrieve_data.iloc[:,:-1].values)
y_datatrieve = datatrieve_data.iloc[:,8].values

#kc2 dataset X and y
X_kc2 = preprocessing.scale(kc2_data.iloc[:,:-1].values)
y_kc2 = kc2_data.iloc[:,21].values

skf = StratifiedKFold(n_splits=5)

### auxiliary methods
def euclidianDistance(v1, v2):
    return math.sqrt(sum(pow((v1 - v2),2)))
    
def getNearestNeighbors(instance, dataset, K):
    neighbours = []
    for i in range(len(dataset)):
        neighbours.append((i, euclidianDistance(instance, dataset[i,:])))
        
    neighbours = np.array(neighbours, dtype=[('index',int),('distance',float)])   
    

    return np.sort(neighbours, order='distance')[0:K]

def getFrequencyPredictedClass(neighbours, dataset_y):
    classesVotes = []
    for i in range(len(neighbours)):
        (i,d) = neighbours[i]
        classesVotes.append(dataset_y[i])
    
    return pd.value_counts(classesVotes).keys()[0]

def getWeightedPredictedClass(neighbours, dataset_y):
    weightedVotes = {}
    for i in range(len(neighbours)):
        (i,d) = neighbours[i]
        neighbor_class = dataset_y[i] 
        if neighbor_class in weightedVotes:
            if d != 0:
                weightedVotes[neighbor_class] += 1/(math.pow(d,2))
        else:
             if d != 0:
                 weightedVotes[neighbor_class] = 1/(math.pow(d,2))
                 
    return sorted(weightedVotes.items(), key=operator.itemgetter(1), reverse=True)[0][0]
    
    
### auxiliary methods end

  
for train_indexes, test_indexes in skf.split(X_kc2,y_kc2):
     print("TRAIN:", X_kc2[train_indexes], "TEST:", y_kc2[test_indexes], "\n\n\n -----------------")
     os.system("pause")
     
#for train, test in kf.split(dataset_datatrieve):
#    print("%s %s" % (train, test))
#kf.split(dataset_datatrieve)