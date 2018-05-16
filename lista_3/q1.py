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


datatrieve_data = pd.read_csv("dataset/datatrieve.csv",header=None)

#datatrieve dataset X and y
X_datatrieve = preprocessing.scale(datatrieve_data.iloc[:,:-1].values)
y_datatrieve = datatrieve_data.iloc[:,8].values

#
#datatrieve_data = pd.read_csv("dataset/datatrieve.csv",header=None)
#X_datatrieve = preprocessing.scale(datatrieve_data.iloc[:,:-1].values)
#y_datatrieve = datatrieve_data.iloc[:,8].values


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


    eig_vals, eig_vecs = np.linalg.eig(np.matmul(np.linalg.inv(Sw),Sb))
    
    eig_map = sortEigenComponents(eig_vals,eig_vecs)
    
    return eig_map  
    
#LDA(X_kc2, y_kc2)
#LDA(X_iris, y_iris)
#LDA(X_cm1, y_cm1)



def buildProjectionMatrix(eig_map, n_components):
    projection_matrix = []

    for i in range(n_components):
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

#components = PCA(X_iris)
#components = PCA(X_cm1)
#components_lda = LDA(X_cm1,y_cm1)
#
#projectionMatrix = buildProjectionMatrix(components, 3)

#X_iris_projected = np.matmul(X_iris,projectionMatrix)


#experiment and plot
import seaborn as sns
import pandas as pd

def barPlot(data):
    df = pd.DataFrame(data, columns=['components','precision'])
    sns.set_style("whitegrid")
    g = sns.barplot(x="components", y="precision", data=df)
    g.set_ylim(0, 1)


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

from sklearn.neighbors import KNeighborsClassifier
knn3 = KNeighborsClassifier(n_neighbors=3)


def runKnnAndGetScores(X_train, y_train, X_test, y_test):
    #train
    knn3.fit(X_train, y_train)
    #accuracies
    score = knn3.score(X_test, y_test)
    print('oi, matrix de precisão' )
    print(score)
#    os.system("pause")
    return []

def runStratifiedKFoldAndPlotResult(X,y):
    n_attributes = len(X[0])
    print(n_attributes)
    for train_indexes, test_indexes in skf.split(X,y):
        #getting components from train set
        components_pca = PCA(X[train_indexes])
        components_lda = LDA(X[train_indexes],y[train_indexes])
        print('\n\n\n\n\n')
        print('queeeee')
        print(components_pca)
        print(components_lda)
#        os.system("pause")
#        #for each quantity of attribute
#        
        for c_v in range(1,n_attributes+1):
            print(c_v)
            projectionMatrix_pca = buildProjectionMatrix(components_pca, c_v)
            projectionMatrix_lda = buildProjectionMatrix(components_lda, c_v)
            
            print(projectionMatrix_lda)
            print(projectionMatrix_pca)

            
#            os.system("pause")
            
            projected_X_train_pca  = np.matmul(X[train_indexes],projectionMatrix_pca)
            projected_X_train_lda =  np.matmul(X[train_indexes],projectionMatrix_lda)
            
            projected_X_test_pca  = np.matmul(X[test_indexes],projectionMatrix_pca)
            projected_X_test_lda =  np.matmul(X[test_indexes],projectionMatrix_lda)

            
#            X_projeced_pca = np.matmul(X_iris,projectionMatrix_pca)
#            X_projeced_lda = np.matmul(X_iris,projectionMatrix_lda)
            
            print('PCA')
            runKnnAndGetScores(projected_X_train_pca, y[train_indexes], projected_X_test_pca, y[test_indexes])
            print('LDA')
            runKnnAndGetScores(projected_X_train_lda, y[train_indexes], projected_X_test_lda, y[test_indexes])

            os.system("pause")
#            
            #fazer aqui uma função para pegar a matrix de projeção e o X, treinar e pegar o score. depois colocar em um array e tirar a média
            
#        print(len(X[train_indexes][0]))
            
        
        
#runStratifiedKFoldAndPlotResult(X_iris,y_iris)
#runStratifiedKFoldAndPlotResult(X_kc2,y_kc2)
#runStratifiedKFoldAndPlotResult(X_datatrieve,y_datatrieve)

l = range(1,12)
print(l)

#mockedDataset = [[1,0.5],[2,0.6],[3,0.7],[3,0.8],[4,0.9],[5,0.556], [6,0.5],[7,0.6],[8,0.7],[9,0.8],[10,0.9],[11,0.556]]
#barPlot(mockedDataset)

#df = pd.DataFrame(mockedDataset, columns=['components','precision'])
#print(df)
#
#
# 
#     
##plot
#sns.set_style("whitegrid")
##tips = sns.load_dataset("tips")
##print(tips.head())
##ax = sns.barplot(x="day", y="total_bill", data=tips)
#g = sns.barplot(x="components", y="precision", data=df)
#g.set_ylim(0, 1)