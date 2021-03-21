# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import global_variables

from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MaxAbsScaler

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, './bd.xlsx')

def load_data(output,variables,filepath=filename):
    data= pd.read_excel(open(filepath, 'rb'))

    data.replace('sem dados', float('nan'),inplace=True)
    data.replace('x', float('nan'),inplace=True)
    data.fillna(float('nan'), inplace=True)

    dataset = pd.concat([data.loc[:,variables],data.loc[:,output]], axis=1)

    # To drop every line that has a NaN value on the TARGET column
    dataset.dropna(how="any",subset=dataset.columns[[-1]],inplace=True)

    if(output==global_variables.outputs[1]):                          # apenas dos graus 1 a 6; com remoção de sub-graus
      dataset = dataset[dataset["classificação clavien-dindo"] != 0]
      dataset = dataset[dataset["classificação clavien-dindo"] != 7]
      dataset.loc[dataset["classificação clavien-dindo"] == 4] = 3
      dataset.loc[dataset["classificação clavien-dindo"] == 5] = 4
      dataset.loc[dataset["classificação clavien-dindo"] == 6] = 4

    if(output==global_variables.outputs[2]):                          # UCI
      dataset.loc[dataset["dias na UCI"] <= 1] = 0
      dataset.loc[(dataset["dias na UCI"] <= 2) & (dataset["dias na UCI"] > 1)] = 1
      dataset.loc[dataset["dias na UCI"] > 2] = 2

    if(output==global_variables.outputs[6]):                          # IPOP
      dataset.loc[dataset["dias no  IPOP"] <= 7] = 0
      dataset.loc[(dataset["dias no  IPOP"] <= 10) & (dataset["dias no  IPOP"] > 7)] = 1
      dataset.loc[(dataset["dias no  IPOP"] <= 20) & (dataset["dias no  IPOP"] > 10)] = 2
      dataset.loc[dataset["dias no  IPOP"] > 20] = 3
    
    if(output==global_variables.outputs[5]):                          # NAS
      dataset.loc[dataset["total pontos NAS"] <= 60] = 0
      dataset.loc[dataset["total pontos NAS"] > 60] = 1
    
    if(output==global_variables.outputs[4]):
      # Correct input inconsistency
      dataset = dataset.loc[dataset["tempo decorrido após cirurgia (óbito)_até 1 ano"] != 0]

    y =  dataset.iloc[:,-1]

    X = dataset.iloc[:,0:-1]

    if(variables==global_variables.numericas):
        y = y.astype(np.float)
        X = X.astype(np.float)
        
    # Missing values imputation, on X (using KNN)
    imp = KNNImputer(n_neighbors=15)
    X = imp.fit_transform(X)

    #Normalização
    scaler = MaxAbsScaler() 
    X = scaler.fit_transform(X)

    return dataset,X,y

def get_X(variables,filepath=filename):
    data= pd.read_excel(open(filepath, 'rb'))

    data.replace('sem dados', float('nan'),inplace=True)
    data.replace('x', float('nan'),inplace=True)
    data.fillna(float('nan'), inplace=True)

    X = data.loc[:,variables]

    # To drop every line that has a NaN value on the TARGET column
    # dataset.dropna(how="any",subset=dataset.columns[[-1]],inplace=True)

    # y =  dataset.iloc[:,-1]

    # X = dataset.iloc[:,0:-1]

    # if(variables==global_variables.numericas):
    #     y = y.astype(np.float)
    #     X = X.astype(np.float)
        
    # # Missing values imputation, on X (using KNN)
    # imp = KNNImputer(n_neighbors=15)
    # X = imp.fit_transform(X)

    #Normalização
    # scaler = MaxAbsScaler() 
    # X = scaler.fit_transform(X)

    return X