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

#certo
baloons_data = pd.read_csv("dataset/q2/baloons-adult-strech.csv",header=None)
#kc2_data = pd.read_csv("dataset/kc2.csv",header=None)

#datatrieve dataset X and y
X_baloons = baloons_data.iloc[:,:-1].values
y_baloons = baloons_data.iloc[:,4].values


balance_data = pd.read_csv("dataset/q2/balance-scale.csv",header=None)
#kc2_data = pd.read_csv("dataset/kc2.csv",header=None)

#datatrieve dataset X and y
X_balance = balance_data.iloc[:,1:5].values
y_balance= balance_data.iloc[:,0].values


#testando

#kc2 dataset X and y
#X_kc2 = preprocessing.scale(kc2_data.iloc[:,:-1].values)
#y_kc2 = kc2_data.iloc[:,21].values

skf = StratifiedKFold(n_splits=5)

## auxiliary methods
#def euclidianDistance(v1, v2):
#    return math.sqrt(sum(pow((v1 - v2),2)))
#    
def getNeighbors(instance, X_train_set, matrix, y_train_set):
    neighbours = []
    for i in range(len(X_train_set)):
        neighbours.append((i, VDM(instance, X_train_set[i,:],matrix,y_train_set)))
        
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
    
K_values_list = [1,2,3,5,7,9,11,13,15]
    
#### knns
def calculateAcurracy(y_true, y_pred):
    return accuracy_score(y_true, np.array(y_pred))
    
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
    
            
def runKNNS(instance, X_train, matrix ,y_train):
    
    neighbours = getNeighbors(instance, X_train, matrix ,y_train)
    
    k_predict_frequency =  {}
    k_predict_weight =  {}
    
    for k in K_values_list:
        k_predict_frequency[k] = getFrequencyPredictedClass(neighbours[0:k], y_train)
        k_predict_weight[k] = getWeightedPredictedClass(neighbours[0:k], y_train)
        
    
    return k_predict_frequency, k_predict_weight
    
def runKNNAndReturnAcurracies(X_train, y_train, X_test,y_test, matrix):
    frequency_predictions =  {}
    weight_predictions =  {}

#    para cada cada no conjunto de testes...
    for index, t in enumerate (X_test):
        predicted_frequency, predicted_weight = runKNNS(t, X_train, matrix, y_train)
#        print(predicted_frequency)
#        print(predicted_weight)
        frequency_predictions[index] = predicted_frequency
        weight_predictions[index] = predicted_weight
    
#    print(len(frequency_predictions))
#    print(len(weight_predictions))
#    print(len(y_test))
#    print(y_test)
#    print(frequency_predictions)
    return [calculateKAcurracyScore(y_test, frequency_predictions),calculateKAcurracyScore(y_test, weight_predictions) ]
  
#######


####### VDM     
def VDM(a,b, matrix, y_train_set):
    distanceVDM = 0
    for i in range(len(a)):
        distanceVDM += vdm(i,a[i],b[i], matrix, y_train_set)       
        
    return math.sqrt(distanceVDM)
    
def vdm(i,ai,bi, matrix,y_train_set):
    classes = np.unique(y_train_set)
    distance = 0

    for c in classes:
        distance += math.pow(( matrix[''.join([str(i),str(ai),str(c)])] - matrix[''.join([str(i),str(bi),str(c)])]),2)   
    
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
##############################

####plot
import matplotlib.pyplot as plt


def barPlot(axis_x, axis_y, text):
    bar_width = 0.3
    plt.figure(1)
    plt.xlabel('K')
    plt.ylabel('precision')
    plt.title(text)
    plt.grid()
    plt.xlim(0,17)
    plt.xticks(np.arange(min(axis_x), max(axis_x)+1, 1.0))
    
    plt.bar(np.asarray(axis_x),axis_y ,bar_width,color='r',label='Accuracy')
    plt.show()
####
 
def precisionMedia(precision_matrix):
    print(precision_matrix)
    l = len(precision_matrix);
    return np.array(precision_matrix).sum(axis=0)/l  

def runCrossValidationAndPlotResult(X, y, database_name):
    allFrequencies= []
    allWeight = []

    for train_indexes, test_indexes in skf.split(X,y):
        matrix = buildProbabilityMatrix(X[train_indexes], y[train_indexes])
        frequency, weight = runKNNAndReturnAcurracies(X[train_indexes],y[train_indexes], X[test_indexes],y[test_indexes],matrix )
         
        allFrequencies.append(frequency)
        allWeight.append(weight)

    barPlot(K_values_list, precisionMedia(allFrequencies), 'KNN' +' - ' +database_name )
    barPlot(K_values_list, precisionMedia(allWeight), 'KNN com peso' + ' - ' +  database_name)
    
runCrossValidationAndPlotResult(X_baloons, y_baloons, 'BALOONS')
runCrossValidationAndPlotResult(X_balance,y_balance, 'BALANCE')