# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:10:23 2018

@author: Guilherme
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

cm1_data = pd.read_csv("dataset/cm1.csv", header=None)
X_cm1 = preprocessing.scale(cm1_data.iloc[:,:-1].values)
y_cm1 = cm1_data.iloc[:,21].values

#datatrieve dataset X and y
datatrieve_data = pd.read_csv("dataset/datatrieve.csv",header=None)
X_datatrieve = preprocessing.scale(datatrieve_data.iloc[:,:-1].values)
y_datatrieve = datatrieve_data.iloc[:,8].values


def sortEigenComponents(eig_vals, eig_vecs):
    eig_map = []

    for i, eig_val in enumerate(eig_vals):
        eig_map.append((eig_val,eig_vecs[i].tolist()))
        

    eig_map = sorted(eig_map, key=lambda my_tuple: my_tuple[0], reverse=True)
    return eig_map
    
    

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



#experiment and plot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def barPlot(data, plot_title):
    
    df = pd.DataFrame(data, columns=['components','precision'])
    #colors
    norm = plt.Normalize(df['precision'].values.min(), df['precision'].values.max())
    colors = plt.cm.Reds(norm(df['precision']))
    ##
    sns.set_style("whitegrid")
    plt.figure(figsize=(8,5))
    
    g = sns.barplot(x="components", y="precision", data=df, palette=colors)
    g.set_title(plot_title)
    g.set_ylim(0, 1)
    plt.show()
    plt.clf()



from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)


def runKnnAndGetScores(X_train, y_train, X_test, y_test):
    #train
    knn.fit(X_train, y_train)
    #accuracies
    score = knn.score(X_test, y_test)
#    os.system("pause")
    return score

def buildPDFramePCA(mean_array, max_range):
    
    pd_format = []
    for i in range(1,max_range+1):
        if i != (max_range):
            pd_format.append([i,mean_array[i-1]])
        else:
            pd_format.append(['original',mean_array[i-1]])
        
    df = pd.DataFrame(pd_format, columns=['components','precision'])
    return df


#########################
def runStratifiedKFoldAndGetFrame(X,y):
    n_attributes = len(X[0])    

    precision_matrix_pca = []

    for train_indexes, test_indexes in skf.split(X,y):
        #getting components from train set
        components_pca = PCA(X[train_indexes])
       
        iteration_precision_line_pca = []

        #for each quantity of attribute
        for c_v in range(1,n_attributes):
            #build the projection matrix of the components
            projectionMatrix_pca = buildProjectionMatrix(components_pca, c_v)
            #projected train set
            projected_X_train_pca  = np.matmul(X[train_indexes],projectionMatrix_pca)
            #projected test set
            projected_X_test_pca  = np.matmul(X[test_indexes],projectionMatrix_pca)
            
            score_pca = runKnnAndGetScores(projected_X_train_pca, y[train_indexes], projected_X_test_pca, y[test_indexes])
            
            iteration_precision_line_pca.append(score_pca)
            
                    
        #appending the original dataset precision
        original_dataset_score = runKnnAndGetScores(X[train_indexes], y[train_indexes], X[test_indexes], y[test_indexes])
        iteration_precision_line_pca.append(original_dataset_score)
        #####
        
        precision_matrix_pca.append(iteration_precision_line_pca)
    
        
        
    return buildPDFramePCA(np.mean(np.array(precision_matrix_pca),axis=0),n_attributes)

    
def runExperimentAndPlot(X,y, dataset_name):
    pd_frame_pca = runStratifiedKFoldAndGetFrame(X,y)
    barPlot(pd_frame_pca, dataset_name + ' PCA')



runExperimentAndPlot(X_cm1,y_cm1, 'CM1')
runExperimentAndPlot(X_datatrieve,y_datatrieve,'DATATRIEVE')
