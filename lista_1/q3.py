# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:26:57 2018

@author: guilherme.santos
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
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


#####german
german_data = pd.read_csv("dataset/q3/german.csv",header=None)

#datatrieve dataset X and y
X_german = german_data.iloc[:,:-1]
y_german = german_data.iloc[:,20].values

# normalizando dados numéricos
X_german[[1,4,7,10,12,15,17]] = preprocessing.scale(X_german[[1,4,7,10,12,15,17]])
X_german = X_german.values

################## crx dataset preprocessing
crx_data = pd.read_csv("dataset/q3/crx.csv",header=None)
#kc2_data = pd.read_csv("dataset/kc2.csv",header=None)

#datatrieve dataset X and y
X_crx = crx_data.iloc[:,:-1]

#X_crx[[1,2, 7,10, 13,14]] = X_crx[[1,2, 7,10, 13,14]].replace('?',np.NaN)
#replacing missing data with NaN for normalize
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
X_crx[[1,2, 7,10, 13,14]] = imputer.fit_transform(X_crx[[1,2, 7,10, 13,14]].replace('?',np.NaN))

#normalizing numerical columns
X_crx[[1,2, 7,10, 13,14]] = preprocessing.scale(X_crx[[1,2, 7,10, 13,14]])

X_crx = X_crx.values
y_crx = crx_data.iloc[:,15].values

#########


skf = StratifiedKFold(n_splits=5)


def getNeighbors(instance, X_train_set, matrix, y_train_set):
    neighbours = []
    for i in range(len(X_train_set)):
        neighbours.append((i, HVDM(instance, X_train_set[i,:],matrix,y_train_set)))
        
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
       
        frequency_predictions[index] = predicted_frequency
        weight_predictions[index] = predicted_weight
        

    return [calculateKAcurracyScore(y_test, frequency_predictions),calculateKAcurracyScore(y_test, weight_predictions) ]

            
##### HVDM            
    
def euclidianDistance(v1, v2):
    return math.sqrt(pow((v1 - v2),2))
    
def HVDM(a,b, matrix, y_train_set):
    distanceHVDM = 0
    for i in range(len(a)):
        if isMissing(a[i]) and isMissing(a[2]):
            distanceHVDM += 0
        elif isMissing(a[i]) or isMissing(a[2]):
            distanceHVDM += 1    
        elif isNumber(a[i]) and isNumber(b[i]):
            distanceHVDM += euclidianDistance(a[i],b[i])
        else:

            distanceHVDM += math.pow(vdm(i,a[i],b[i], matrix, y_train_set),2)   


    return math.sqrt(distanceHVDM)
    
def vdm(i,ai,bi, matrix,y_train_set):
    classes = np.unique(y_train_set)
    distance = 0

    for c in classes:
        aKey = ''.join([str(i),str(ai),str(c)])
        bKey = ''.join([str(i),str(bi),str(c)])
        #se não existe na matrix é pq a probabilidade é 0
          
        if(aKey in matrix and bKey in matrix): 
            distance += math.pow(( matrix[aKey] - matrix[bKey]),2)
        else:
            print('não tem')
            
    return distance  


def isNumber(number):
    
    return str(number).lstrip('-').replace('.','',1).isnumeric()
 
def isMissing(data):
    return data == '?'
    
def buildProbabilityMatrix(X_train_set, y_train_set):
    classes = np.unique(y_train_set)
    n_attributes = len(X_train_set[0,:])
    matrix = {}
    
    for attr_index in range(n_attributes):
        currentColumn = X_train_set[:,attr_index]


        if isNumber(str(X_train_set[0,:][attr_index])):

            continue

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

    l = len(precision_matrix);
    return np.array(precision_matrix).sum(axis=0)/l  

def runCrossValidationAndPlotResult(X, y, database_name):
    allFrequencies= []
    allWeight = []

    for train_indexes, test_indexes in skf.split(X,y):
        
        matrix = buildProbabilityMatrix(X[train_indexes], y[train_indexes])
#        print(matrix)
        frequency, weight = runKNNAndReturnAcurracies(X[train_indexes],y[train_indexes], X[test_indexes],y[test_indexes],matrix )
         
        print('TERMINOU A ITERAÇÃO')
        print(matrix)
        allFrequencies.append(frequency)
        allWeight.append(weight)
        
    barPlot(K_values_list, precisionMedia(allFrequencies), 'KNN' +' - ' +database_name )
    barPlot(K_values_list, precisionMedia(allWeight), 'KNN com peso' + ' - ' +  database_name)
        
runCrossValidationAndPlotResult(X_german, y_german, 'GERMAN')         
#runCrossValidationAndPlotResult(X_post_operative, y_post_operative, 'POST OPERATIVE')
runCrossValidationAndPlotResult(X_crx, y_crx, 'CRX')
 