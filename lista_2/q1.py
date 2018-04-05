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

knnClassifier = KNeighborsClassifier(n_neighbors=1)
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
    
#    print(random_prototypes_X)
#    print(random_prototypes_y)
    return [np.array(random_prototypes_X), np.array(random_prototypes_y)] 

def learningRateByEpoch(learning_rate, currentEpoch, totalEpochs):
    return learning_rate*(1.0-currentEpoch/float(totalEpochs))
    
def lvq(X_dataset, y_dataset, n_prototypes, learning_rate, epochs_quantity):
    [selected_prototypes_X, selected_prototypes_y] = selectRandomProtypes(n_prototypes, X_dataset, y_dataset) 
    knnClassifier.fit(selected_prototypes_X,selected_prototypes_y)
#    
    print(selected_prototypes_X)
#    print(selected_prototypes_y)
    for epoch in range(epochs_quantity):
        
        for i,x in enumerate(X_dataset):
        
            print(x)
            print(knnClassifier.predict([x],)[0])
#            print(y_dataset[i])
#            nearestPrototype
#            print(knnClassifier.predict([x])[0] == y_dataset[i])
#           
#            if (knnClassifier.predict([x])[0] != y_dataset[i]):
                
                
            os.system("pause")
            
            
        print(learningRateByEpoch(learning_rate, epoch, epochs_quantity))
    

    return [selected_prototypes_X, selected_prototypes_y]
    
lvq(X_datatrieve, y_datatrieve, 5, 0.3, 20)