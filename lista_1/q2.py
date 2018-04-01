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
from sklearn.metrics import accuracy_score


baloons_data = pd.read_csv("dataset/q2/baloons-adult-strech.csv",header=None)
#kc2_data = pd.read_csv("dataset/kc2.csv",header=None)

#datatrieve dataset X and y
X_baloons = baloons_data.iloc[:,:-1].values
y_baloons = baloons_data.iloc[:,4].values

#kc2 dataset X and y
#X_kc2 = preprocessing.scale(kc2_data.iloc[:,:-1].values)
#y_kc2 = kc2_data.iloc[:,21].values

skf = StratifiedKFold(n_splits=5)

## auxiliary methods
#def euclidianDistance(v1, v2):
#    return math.sqrt(sum(pow((v1 - v2),2)))
#    
def getNearestNeighbors(instance, X_train_set, K, matrix, y_train_set):
    neighbours = []
    for i in range(len(X_train_set)):
        neighbours.append((i, VDM(instance, X_train_set[i,:],matrix,y_train_set)))
        
    neighbours = np.array(neighbours, dtype=[('index',int),('distance',float)])   
    
    return np.sort(neighbours, order='distance')[0:K]

def getFrequencyPredictedClass(neighbours, dataset_y):
    classesVotes = []
    for i in range(len(neighbours)):
        (i,d) = neighbours[i]
        classesVotes.append(dataset_y[i])
    
    return pd.value_counts(classesVotes).keys()[0]

#def getWeightedPredictedClass(neighbours, dataset_y):
#    weightedVotes = {}
#    for i in range(len(neighbours)):
#        (i,d) = neighbours[i]
#        neighbor_class = dataset_y[i] 
#        if neighbor_class in weightedVotes:
#            if d != 0:
#                weightedVotes[neighbor_class] += 1/(math.pow(d,2))
#        else:
#             if d != 0:
#                 weightedVotes[neighbor_class] = 1/(math.pow(d,2))
#                 
#    return sorted(weightedVotes.items(), key=operator.itemgetter(1), reverse=True)[0][0]
#    

def VDM(a,b, matrix, y_train_set):
    distanceVDM = 0
    for i in range(len(a)):
        distanceVDM += vdm(i,a[i],b[i], matrix, y_train_set)       
        
    return math.sqrt(distanceVDM)
    
def vdm(i,ai,bi, matrix,y_train_set):
    classes = np.unique(y_train_set)
    distance = 0
#    print(ai)
#    print(bi)
    for c in classes:
        distance += math.pow(( matrix[''.join([str(i),str(ai),str(c)])] - matrix[''.join([str(i),str(bi),str(c)])]),2)
#        print( matrix[''.join([str(i),str(ai),str(c)])])
#        print( matrix[''.join([str(i),str(bi),str(c)])])
#        print(distance)
#        os.system("pause")
    
    return distance  

    
def buildProbabilityMatrix(X_train_set, y_train_set):
    classes = np.unique(y_train_set)
    n_attributes = len(X_baloons[0,:])
    
    matrix = {}
    
    for attr_index in range(n_attributes):
        currentColumn = X_train_set[:,attr_index]
        
        for word in np.unique(currentColumn):
            f = countElements(word,currentColumn)
            for c in classes:
                fc = countElementsWithClass(word, c,currentColumn, y_train_set)
                matrix[''.join([str(attr_index),str(word),str(c)])] = (fc/f)

    return matrix

def countElements(element, array):
    count = 0
    for l in array:
        if element == l:
            count +=1
    return count
    
def countElementsWithClass(element, c, array, y):
    count = 0
    for i in range(len(array)):
        if array[i] == element and y[i] == c:
            count +=1  
        
    return count
    
### auxiliary methods end
for train_indexes, test_indexes in skf.split(X_baloons,y_baloons):
    matrix = buildProbabilityMatrix(X_baloons[train_indexes], y_baloons[train_indexes])
#    print(VDM(X_baloons[train_indexes][0,:], X_baloons[train_indexes][5,:],matrix, y_baloons[train_indexes]))
#     print("TRAIN:", X_baloons[train_indexes], "TEST:", y_baloons[test_indexes], "\n\n\n -----------------")
#    print(getNearestNeighbors(X_baloons[test_indexes][0,:], X_baloons[train_indexes], 5, matrix, y_baloons[train_indexes]))
    neighbours = getNearestNeighbors(X_baloons[test_indexes][2,:], X_baloons[train_indexes], 5, matrix, y_baloons[train_indexes])    
    print(getFrequencyPredictedClass(neighbours, y_baloons[train_indexes]))
    os.system("pause")
     
#for train, test in kf.split(dataset_datatrieve):
#    print("%s %s" % (train, test))
#kf.split(dataset_datatrieve)