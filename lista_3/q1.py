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


a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a.mean(axis=1)


classes = np.unique(y_kc2).tolist()


Sw = 0
Sb = 0

total_size = 0

for c in classes:
   
    print(len(np.where(y_kc2 == c)[0]))
    print(c)
    

print(total_size)
print(len(y_kc2))    
os.system("pause")