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
    
def getNeighbors(instance, dataset):
    neighbours = []
    for i in range(len(dataset)):
        neighbours.append((i, euclidianDistance(instance, dataset[i,:])))
        
    neighbours = np.array(neighbours, dtype=[('index',int),('distance',float)])   
    

#    return np.sort(neighbours, order='distance')[0:K]
    return np.sort(neighbours, order='distance')


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
             else: 
                 weightedVotes[neighbor_class] = 0

    
    return sorted(weightedVotes.items(), key=operator.itemgetter(1), reverse=True)[0][0]
    
    
## auxiliary methods end
K_values_list = [1,2,3,5,7,9,11,13,15]

def runKNNAndReturnAcurracies(X_train, y_train, X_test,y_test):
#    neighbours = getNeighbors()
    
    frequency_predictions =  {}
    weight_predictions =  {}

#    para cada cada no conjunto de testes...
    for index, t in enumerate (X_test):
#        print(t)
#        print(index)
        predicted_frequency, predicted_weight = runKNNS(t, X_train, y_train)
        frequency_predictions[index] = predicted_frequency
        weight_predictions[index] = predicted_weight
#        print(predicted_frequency)
#        print(predicted_weight)
#        
        
        
#    np.sort(neighbours, order='distance')[0:K]
#    print(frequency_predictions)
#    print(weight_predictions)
#    print(y_test)
#    print(len(y_test))
#    os.system("pause")
    
    
    #construindo array de valores retornado por cada K
#    y1f, y2f, y3f, y4f, y5f, y7f, y9f, y11f, y13f, y15f = [],[],[],[],[],[],[],[],[],[]
#    #populando os arrays predizidos
#    for index, t in enumerate(X_test):
#        #frequency
#        
#        y1f.append (frequency_predictions[index][1])
#        y2f.append (frequency_predictions[index][2])
#        y3f.append (frequency_predictions[index][3])
#        y4f.append (frequency_predictions[index][4])
#        y5f.append (frequency_predictions[index][5])
#        y7f.append (frequency_predictions[index][7])
#        y9f.append (frequency_predictions[index][9])
#        y11f.append (frequency_predictions[index][11])
#        y13f.append (frequency_predictions[index][13])
#        y15f.append (frequency_predictions[index][15])
    
    
    print(calculateKAcurracyScore(y_test, frequency_predictions))
    print(calculateKAcurracyScore(y_test, weight_predictions))

    os.system("pause")
    return []


def calculateKAcurracyScore(y_true, predictions):
    
    y1, y2, y3, y5, y7, y9, y11, y13, y15 = [],[],[],[],[],[],[],[],[]
    #populando os arrays predizidos
    for index, t in enumerate(predictions):
        #frequency
        
        y1.append (predictions[index][1])
        y2.append (predictions[index][2])
        y3.append (predictions[index][3])
        y5.append (predictions[index][5])
        y7.append (predictions[index][7])
        y9.append (predictions[index][9])
        y11.append (predictions[index][11])
        y13.append (predictions[index][13])
        y15.append (predictions[index][15])
        
        
    return [calculateAcurracy(y_true, y1),calculateAcurracy(y_true, y2), calculateAcurracy(y_true, y3), calculateAcurracy(y_true, y5), calculateAcurracy(y_true, y7), calculateAcurracy(y_true, y9), calculateAcurracy(y_true, y11), calculateAcurracy(y_true, y13),calculateAcurracy(y_true, y15)] 
    
def calculateAcurracy(y_true, y_pred):
    return accuracy_score(y_true, np.array(y_pred))
    
def runKNNS(instance, X_train,y_train):
    neighbours = getNeighbors(instance, X_train)
    
    k_predict_frequency =  {}
    k_predict_weight =  {}
    
    for k in K_values_list:
        k_predict_frequency[k] = getFrequencyPredictedClass(neighbours[0:k], y_train)
        k_predict_weight[k] = getWeightedPredictedClass(neighbours[0:k], y_train)
        
    
#    print(k_predict_frequency)
#    print(k_predict_weight)
#    os.system("pause")
    return k_predict_frequency, k_predict_weight



  
for train_indexes, test_indexes in skf.split(X_kc2,y_kc2):
#     print("TRAIN:", X_kc2[train_indexes], "TEST:", y_kc2[test_indexes], "\n\n\n -----------------")
     rrrr(X_kc2[train_indexes],y_kc2[train_indexes], X_kc2[test_indexes],y_kc2[test_indexes], )
     

     os.system("pause")
     

     
     
