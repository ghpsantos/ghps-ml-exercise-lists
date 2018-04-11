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

#datatrieve_data = pd.read_csv("dataset/datatrieve.csv",header=None)
kc2_data = pd.read_csv("dataset/kc2.csv",header=None)
X_kc2 = preprocessing.scale(kc2_data.iloc[:,:-1].values)
y_kc2 = kc2_data.iloc[:,21].values

cm1_data = pd.read_csv("dataset/cm1.csv", header=None)
X_cm1 = preprocessing.scale(cm1_data.iloc[:,:-1].values)
y_cm1 = cm1_data.iloc[:,21].values

#datatrieve dataset X and y
#X_datatrieve = preprocessing.scale(datatrieve_data.iloc[:,:-1].values)
#y_datatrieve = datatrieve_data.iloc[:,8].values

#kc2 dataset X and y


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
        selected_indexes = random.sample(np.asarray(np.where(y_dataset==c)).tolist()[0],n_prototypes)
        for i in selected_indexes:
            random_prototypes_X.append(np.asarray(X_dataset[i]).tolist())
            random_prototypes_y.append(np.asarray(y_dataset[i]).tolist())
            
#    print(random_prototypes_X)
#    print(random_prototypes_y)
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
    
def lvq21 (X_dataset, y_dataset, n_prototypes, learning_rate, epochs_quantity, w):
    [P_X,p_y] = lvq1(X_dataset, y_dataset, n_prototypes, learning_rate, epochs_quantity)
    knn2Classifier.fit(P_X, p_y)

    for epoch in range(epochs_quantity):
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

def lvq3 (X_dataset, y_dataset, n_prototypes, learning_rate, epochs_quantity, w, epsilon):
    [P_X,p_y] = lvq1(X_dataset, y_dataset, n_prototypes, learning_rate, epochs_quantity)
    knn2Classifier.fit(P_X, p_y)

    for epoch in range(epochs_quantity):
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
                    #if all classes are 
                    elif(neigh_1_class == neigh_2_class == y_dataset[i]):
                        P_X[index_neigh_1] += epsilon*(learningRateByEpoch(learning_rate, epoch, epochs_quantity)*(x - P_X[index_neigh_1]))
                        P_X[index_neigh_2] += epsilon*(learningRateByEpoch(learning_rate, epoch, epochs_quantity)*(x - P_X[index_neigh_2]))
        
    return [P_X, p_y]

    
#[T_X_1, T_y_1] = lvq1(X_datatrieve, y_datatrieve, 8, 0.3, 20)
#[T_X_1, T_y_1] = lvq1(X_cm1, y_cm1, 8, 0.3, 20)

#[T_X_2, T_y_2] = lvq21(X_datatrieve, y_datatrieve, 8, 0.3, 20, 0.01)
#[T_X_3, T_y_3] = lvq3(X_kc2, y_kc2, 8, 0.3, 20, 0.01, 0.1)

#[T_X_3, T_y_3] = lvq1(X_cm1, y_cm1, 8, 0.3, 20)

#print(T_X_1)
#print(T_y_1)
#print(T_X_2)
#print(T_y_2)
#print(T_X_3)
#print(T_y_3)


#knnClassifier33 = KNeighborsClassifier(n_neighbors=2)
#knnClassifier33.fit(T_X_1, T_y_1)

#from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

#for train_index, test_index in skf.split(X_datatrieve, y_datatrieve):
    
    
#    print(y_datatrieve[train_index])
#    print(np.where( y_datatrieve == 1))
#    print(len(y_datatrieve[train_index]))
#    os.system("pause")

knn1 = KNeighborsClassifier(n_neighbors=1)
knn3 = KNeighborsClassifier(n_neighbors=3)
n_epochs = 100
lr = 0.4
w = 0.3
epsilon = 0.1 

def precisionMediaAndStd(precision_matrix):
    return [np.mean(np.array(precision_matrix)), np.std(np.array(precision_matrix))]

def runKNNsAndReturnAccuracies(X_train, y_train, X_test,y_test):
    knn1.fit(X_train, y_train)
    knn3.fit(X_train, y_train)
    return [knn1.score(X_test, y_test), knn3.score(X_test, y_test)]
    
def runLVQs(X, y, n_prototypes):
   
    knn1ScoresLVQ1 = []
    knn1ScoresLVQ21 = []
    knn1ScoresLVQ3 = []

    knn3ScoresLVQ1 = []
    knn3ScoresLVQ21 = []
    knn3ScoresLVQ3 = []
    
    # sem utilizar algoritmo de seleção de protótipos
    knn1Scores = []
    knn3Scores = []
    
    for train_indexes, test_indexes in skf.split(X,y):
        print(n_prototypes)
        #for numero de protótipos...
#        print(knn1ScoresLVQ1)
        #creating datasets
        [T_X_1, T_y_1] = lvq1(X, y, n_prototypes, lr, n_epochs)
        [T_X_2, T_y_2] = lvq21(X, y, n_prototypes, lr, n_epochs, w)
        [T_X_3, T_y_3] = lvq3(X, y, n_prototypes,lr, n_epochs, w, epsilon)
        
        #running knns and build the accuracy matrix
        lvq1Scores = runKNNsAndReturnAccuracies(T_X_1,T_y_1, X[test_indexes],y[test_indexes])
        knn1ScoresLVQ1.append(lvq1Scores[0])
        knn3ScoresLVQ1.append(lvq1Scores[1])
        
        lvq21Scores = runKNNsAndReturnAccuracies(T_X_2,T_y_2, X[test_indexes],y[test_indexes])
        knn1ScoresLVQ21.append(lvq21Scores[0])
        knn3ScoresLVQ21.append(lvq21Scores[1])
        
        lvq3Scores = runKNNsAndReturnAccuracies(T_X_3,T_y_3, X[test_indexes],y[test_indexes])
        knn1ScoresLVQ3.append(lvq3Scores[0])
        knn3ScoresLVQ3.append(lvq3Scores[1])
        
        trainScore = runKNNsAndReturnAccuracies(X[train_indexes],y[train_indexes], X[test_indexes],y[test_indexes])
        knn1Scores.append(trainScore[0])
        knn3Scores.append(trainScore[1])
     
        
    return [precisionMediaAndStd(knn1ScoresLVQ1),
            precisionMediaAndStd(knn1ScoresLVQ21),
            precisionMediaAndStd(knn1ScoresLVQ3),
            precisionMediaAndStd(knn1Scores),
            precisionMediaAndStd(knn3ScoresLVQ1),
            precisionMediaAndStd(knn3ScoresLVQ21),
            precisionMediaAndStd(knn3ScoresLVQ3),
            precisionMediaAndStd(knn3Scores)]

def printInFormat(database_name, n_prototipes, precisions):
    print(database_name)
    print('--------------------------------------')
    print('Número de Protótipos: ' + str(n_prototipes))
    print('--------------------------------------')
    print('KNN - 1')
    
    print('LVQ1: ', str(precisions[0][0]) , '±', str(precisions[0][1]), 'σ')
    print('LVQ21: ', str(precisions[1][0]) , '±', str(precisions[1][1]), 'σ')
    print('LVQ3: ', str(precisions[2][0]) , '±', str(precisions[2][1]), 'σ')
    print('TrainSet: ', str(precisions[3][0]) , '±', str(precisions[3][1]), 'σ')

    print('--------------------------------------')
    
    print('KNN - 3')
    print('LVQ1: ', str(precisions[4][0]) , '±', str(precisions[4][1]), 'σ')
    print('LVQ21: ', str(precisions[5][0]) , '±', str(precisions[5][1]), 'σ')
    print('LVQ3: ', str(precisions[6][0]) , '±', str(precisions[6][1]), 'σ')
    print('TrainSet: ', str(precisions[7][0]) , '±', str(precisions[7][1]), 'σ')

    print('--------------------------------------')

#printInFormat(2, [12.2,2,3])

prototype_values = [4,8,16,32]
for p_v in prototype_values:
    #running LVQs and printing the results
    printInFormat('CM1 DATABASE' , p_v, runLVQs(X_cm1, y_cm1, p_v))  
    printInFormat('KC2 DATABASE' , p_v, runLVQs(X_kc2, y_kc2, p_v))  
    
#printInFormat(8, runLVQs(X_cm1, y_cm1, 8))    
#runLVQs(X_cm1, y_cm1, 8)