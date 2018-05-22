# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:55:33 2018

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
    
def buildProjectionMatrix(eig_map, n_components):
    projection_matrix = []

    for i in range(n_components):
        projection_matrix.append(np.transpose([eig_map[i][1]]))
    
    #join arrays by column
    return np.hstack(projection_matrix)
    

#experiment and plot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def barPlot(data, plot_title):
    
    df = pd.DataFrame(data, columns=['components','precision'])
#    print(df)
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

    return score

def buildPDFrameLDA(mean_array, max_range):
    pd_format = []
    for i in range(1,max_range+1):
        if i != (max_range):
            pd_format.append([i,mean_array[i-1]])
        else:
            pd_format.append(['original',mean_array[i-1]])
        
    df = pd.DataFrame(pd_format, columns=['components','precision'])
    return df

def runStratifiedKFoldAndGetFrame(X,y):
    n_classes = len(np.unique(y).tolist())        

    precision_matrix_lda = []


    for train_indexes, test_indexes in skf.split(X,y):
        #getting components from train set
        components_lda = LDA(X[train_indexes],y[train_indexes])
       
        iteration_precision_line_lda = []

        #for each quantity of attribute

        for c_v in range(1,n_classes):
            projectionMatrix_lda = buildProjectionMatrix(components_lda, c_v)
                  
            projected_X_train_lda =  np.matmul(X[train_indexes],projectionMatrix_lda)
            
            projected_X_test_lda =  np.matmul(X[test_indexes],projectionMatrix_lda)

            score_lda = runKnnAndGetScores(projected_X_train_lda, y[train_indexes], projected_X_test_lda, y[test_indexes])
            
            iteration_precision_line_lda.append(score_lda)
            

         #appending the original dataset precision
        original_dataset_score = runKnnAndGetScores(X[train_indexes], y[train_indexes], X[test_indexes], y[test_indexes])
        iteration_precision_line_lda.append(original_dataset_score)
        #####
        
        precision_matrix_lda.append(iteration_precision_line_lda)
        


    
    return buildPDFrameLDA(np.mean(np.array(precision_matrix_lda),axis=0),n_classes)   

def runExperimentAndPlot(X,y, dataset_name):
    pd_frame_pca = runStratifiedKFoldAndGetFrame(X,y)
    barPlot(pd_frame_pca, dataset_name + ' LDA')
    

runExperimentAndPlot(X_cm1,y_cm1, 'CM1')
runExperimentAndPlot(X_datatrieve,y_datatrieve,'DATATRIEVE')

     