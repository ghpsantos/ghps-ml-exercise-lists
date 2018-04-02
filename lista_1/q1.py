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
        predicted_frequency, predicted_weight = runKNNS(t, X_train, y_train)
        frequency_predictions[index] = predicted_frequency
        weight_predictions[index] = predicted_weight
    
    return [calculateKAcurracyScore(y_test, frequency_predictions),calculateKAcurracyScore(y_test, weight_predictions) ]


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
        
    
    return k_predict_frequency, k_predict_weight


import matplotlib.pyplot as plt


def barPlot(axis_x, axis_y, text):
    bar_width = 0.3
    plt.figure(1)
    plt.xlim([0,1])
    plt.xlabel('K')
    plt.ylabel('precision')
    plt.title(text)
    plt.grid()
    plt.xlim(0,17)
    plt.xticks(np.arange(min(axis_x), max(axis_x)+1, 1.0))
    
    plt.bar(np.asarray(axis_x),axis_y ,bar_width,color='r',label='Accuracy')
    plt.show()


def precisionMedia(precision_matrix):
    l = len(precision_matrix);
    return np.array(precision_matrix).sum(axis=0)/l


def runCrossValidationAndPlotResult(X, y, database_name):
    allFrequencies= []
    allWeight = []

    for train_indexes, test_indexes in skf.split(X,y):
         frequency, weight = runKNNAndReturnAcurracies(X[train_indexes],y[train_indexes], X[test_indexes],y[test_indexes] )
         
         allFrequencies.append(frequency)
         allWeight.append(weight)

    barPlot(K_values_list, precisionMedia(allFrequencies), 'KNN' +' - ' +database_name )
    barPlot(K_values_list, precisionMedia(allWeight), 'KNN com peso' + ' - ' +  database_name)
    
runCrossValidationAndPlotResult(X_kc2, y_kc2, 'KC2')
runCrossValidationAndPlotResult(X_datatrieve,y_datatrieve, 'DATATRIEVE')





