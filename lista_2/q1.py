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
import random
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

datatrieve_data = pd.read_csv("dataset/datatrieve.csv",header=None)
kc2_data = pd.read_csv("dataset/kc2.csv",header=None)

#datatrieve dataset X and y
X_datatrieve = preprocessing.scale(datatrieve_data.iloc[:,:-1].values)
y_datatrieve = datatrieve_data.iloc[:,8].values

#kc2 dataset X and y
X_kc2 = preprocessing.scale(kc2_data.iloc[:,:-1].values)
y_kc2 = kc2_data.iloc[:,21].values

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test  = train_test_split(X_datatrieve,y_datatrieve, test_size = 0.2, random_state = 0)

knn1Classifier = KNeighborsClassifier(n_neighbors=1)
#neigh.fit(X_train, y_train)

#print(neigh.kneighbors([X_test[3]],n_neighbors=5))
#skf = StratifiedKFold(n_splits=5)

def euclidianDistance(v1, v2):
    return math.sqrt(sum(pow((v1 - v2),2)))

#n_prototypes gets N of each class in the dataset
def selectRandomProtypes(n_prototypes, X_dataset,y_dataset):
    random_prototypes_X = []
    random_prototypes_y = []
    classes = np.unique(y_dataset)
    for c in classes:
        selected_indexes = random.sample(np.asarray(np.where(y_datatrieve==c)).tolist()[0],n_prototypes)
        for i in selected_indexes:
            random_prototypes_X.append(np.asarray(X_dataset[i]).tolist())
            random_prototypes_y.append(np.asarray(y_dataset[i]).tolist())
    
    return [np.array(random_prototypes_X), np.array(random_prototypes_y)] 

#learning rate in function of time
def learningRateByEpoch(learning_rate, currentEpoch, totalEpochs):
    return learning_rate*(1.0-currentEpoch/float(totalEpochs))
    
def lvq1(X_dataset, y_dataset, n_prototypes, learning_rate, epochs_quantity):
    [selected_prototypes_X, selected_prototypes_y] = selectRandomProtypes(n_prototypes, X_dataset, y_dataset) 
    knn1Classifier.fit(selected_prototypes_X,selected_prototypes_y)
    
    for epoch in range(epochs_quantity):
        for i,x in enumerate(X_dataset):
            #get nearest neighbor
            _ , neighs = knn1Classifier.kneighbors([x])            
            index = neighs[0][0]
            #adjusting prototype
            if (selected_prototypes_y[index] == y_dataset[i]):
                selected_prototypes_X[index] += (learningRateByEpoch(learning_rate, epoch, epochs_quantity)*(x - selected_prototypes_X[index]))
            else:
                selected_prototypes_X[index] -= (learningRateByEpoch(learning_rate, epoch, epochs_quantity)*(x - selected_prototypes_X[index]))

                

    return [selected_prototypes_X, selected_prototypes_y]

knn2Classifier = KNeighborsClassifier(n_neighbors=2)

def isInWindow(distance_1, distance_2, w):
    s = (1-w)/(1+w)
    d1 = distance_1/distance_2
    d2 = distance_2/distance_1
    
    return min(d1,d2) > s
    
def lvq2 (X_dataset, y_dataset, n_prototypes, learning_rate, epochs_quantity, w):
    [P_X,p_y] = lvq1(X_dataset, y_dataset, n_prototypes, learning_rate, epochs_quantity)
    knn2Classifier.fit(P_X, p_y)

    for epoch in range(epochs_quantity):
        print(learningRateByEpoch(learning_rate, epoch, epochs_quantity))
        for i,x in enumerate(X_dataset):
            distances , neighs = knn2Classifier.kneighbors([x])
#            print(neighs)
            index_neigh_1 = neighs[0][0]
            index_neigh_2 = neighs[0][1]
            distance_1 = distances[0][0]
            distance_2 = distances[0][1]
            # if is in window
            if isInWindow(distance_1, distance_2, w):
                    neigh_1_class = p_y[index_neigh_1]
                    neigh_2_class = p_y[index_neigh_2]
                    # if there different classes
                    if (neigh_1_class  != neigh_2_class):
                        #adjusting prototype
                        if( neigh_1_class  == y_dataset[i] ):
                            P_X[index_neigh_1] += (learningRateByEpoch(learning_rate, epoch, epochs_quantity)*(x - P_X[index_neigh_1]))
                            P_X[index_neigh_2] -= (learningRateByEpoch(learning_rate, epoch, epochs_quantity)*(x - P_X[index_neigh_2]))
    
                        elif(neigh_2_class == y_dataset[i]):
                             P_X[index_neigh_1] -= (learningRateByEpoch(learning_rate, epoch, epochs_quantity)*(x - P_X[index_neigh_1]))
                             P_X[index_neigh_2] += (learningRateByEpoch(learning_rate, epoch, epochs_quantity)*(x - P_X[index_neigh_2]))
                          
        
    return [P_X, p_y]

    
[T_X_1, T_y_1] = lvq1(X_datatrieve, y_datatrieve, 8, 0.3, 20)
[T_X_2, T_y_2] = lvq2(X_datatrieve, y_datatrieve, 8, 0.3, 20, 555555555555)

print(T_X_1)
print(T_y_1)
print(T_X_2)
print(T_y_2)
