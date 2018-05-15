# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:10:23 2018

@author: Guilherme
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os


kc2_data = pd.read_csv("dataset/kc2.csv",header=None)
X_kc2 = preprocessing.scale(kc2_data.iloc[:,:-1].values)
y_kc2 = kc2_data.iloc[:,21].values

cm1_data = pd.read_csv("dataset/cm1.csv", header=None)
X_cm1 = preprocessing.scale(cm1_data.iloc[:,:-1].values)
y_cm1 = cm1_data.iloc[:,21].values


iris_data = pd.read_csv("dataset/iris.csv", header=None)
X_iris = preprocessing.scale(iris_data.iloc[:,:-1].values)
y_iris = iris_data.iloc[:,4].values



def sortEigenComponents(eig_vals, eig_vecs):
    eig_map = []

    for i, eig_val in enumerate(eig_vals):
#        print(i)
#        print(eig_val)
        eig_map.append((eig_val,eig_vecs[i].tolist()))
        
        
#    print(eig_vals)
#    print(eig_vecs)
    eig_map = sorted(eig_map, key=lambda my_tuple: my_tuple[0], reverse=True)
    return eig_map
    
def LDA(X,y):
    classes = np.unique(y).tolist()
    Sw = 0
    Sb = 0

    overall_mean_dataset = np.mean(X, axis=0)

    for c in classes:
        indexes = np.where(y == c)[0]
        
        #número de padrões da classe l
        nl = len(indexes)
        #class instances
        classSamples = X[indexes]
        #overall mean classs
        overall_mean_class = np.mean(classSamples, axis=0)
        
#        print(overall_mean_class)
        #Sb calculation
        mean_diff = overall_mean_class - overall_mean_dataset
        Sb = Sb + nl*(np.matmul(np.transpose([mean_diff]), np.array([mean_diff])))
        
        #Sw calculation
        for i in range(nl):
            mean_diff = classSamples[i,:] - overall_mean_class
            
            Sw = Sw + np.matmul(np.transpose([mean_diff]), np.array([mean_diff]));


    eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(Sw),Sb))
    
    eig_map = sortEigenComponents(eig_vals,eig_vecs)
    
    return eig_map  
    
LDA(X_kc2, y_kc2)
#LDA(X_iris, y_iris)
#LDA(X_cm1, y_cm1)



def buildProjectionMatrix(eig_map, n_components):
    projection_matrix = []

    for i in range(n_components):
        print(np.transpose([eig_map[i][1]]))
        projection_matrix.append(np.transpose([eig_map[i][1]]))
    
    #join arrays by column
    return np.hstack(projection_matrix)

########## PCA

def PCA(X):
    #calculate the covariance matrix
    cov_matrix = np.cov(X.T)
    # calculate eigenvalues and vectors
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

    #sort eigens
    eig_map = sortEigenComponents(eig_vals,eig_vecs)
    
    return eig_map

components = PCA(X_iris)
#components = PCA(X_cm1)
projectionMatrix = buildProjectionMatrix(components, 3)


X_iris_projected = np.matmul(X_iris,projectionMatrix)
