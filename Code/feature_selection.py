# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import global_variables
import data_processing

from sklearn.feature_selection import chi2, f_classif, f_regression, VarianceThreshold

def f_select(output=None, p_value=0.05, k=None, var_group=None):
    dataset,X,y = data_processing.load_data(output, var_group)

    lista_index_remover = []
    for col in range(0,len(X[0])):
        if(np.var(list(X[:,col]))==0): # no variance (sempre constante)
            lista_index_remover.append(col)
    X = np.delete(X, lista_index_remover, 1)
    dataset.drop(dataset.columns[lista_index_remover], axis=1, inplace=True)

    if(var_group==global_variables.numericas):
        if(output=="dias na UCI" or output=="classificação clavien-dindo" or output=="total pontos NAS"or output=="dias no IPOP"):
            rank = f_regression(X, y)
        else:
            rank = f_classif(X, y)
    elif(var_group==global_variables.categoricas or var_group==global_variables.binarias):
        if(output=="dias na UCI" or output=="classificação clavien-dindo" or output=="total pontos NAS"or output=="dias no IPOP"):
            rank = f_classif(X, y)
        else:
            rank = chi2(X, y)
    elif(var_group==global_variables.ordinais):
        if(output=="dias na UCI" or output=="classificação clavien-dindo" or output=="total pontos NAS"or output=="dias no IPOP"):
            rank = f_regression(X, y)
        else:
            rank = f_classif(X, y)

    lista = [(dataset.columns[i],rank[0][i],rank[1][i]) for i in range(0,len(X[0]))]
    lista = sorted(lista, key=lambda par: par[1])
    lista.reverse()
    lista = list(filter(lambda x: x[2] <= p_value, lista))
    if(k==None):
        k=len(lista)
    result = []
    for i in range(0,k):
        result.append((lista[i][0],lista[i][1]))
    return result

print("ALL:",len(global_variables.numericas+global_variables.binarias+global_variables.categoricas+global_variables.ordinais))
p=[0.05,0.01,0.005,0.0002]
for val in p:
    print(">>>>> P-VALUE = ",val)
    overall=[]
    for output in global_variables.outputs:
        print("\n#### Feature ranking for the output: ####", output)
        selected_features = []
        p=val
        selected_features+=f_select(output=output, p_value=p, k=None, var_group=global_variables.numericas)
        selected_features+=f_select(output=output, p_value=p, k=None, var_group=global_variables.binarias)
        selected_features+=f_select(output=output, p_value=p, k=None, var_group=global_variables.categoricas)
        selected_features+=f_select(output=output, p_value=p, k=None, var_group=global_variables.ordinais)
        selected_features = sorted(selected_features, key=lambda par: par[1])
        selected_features.reverse()
        result = [x[0] for x in selected_features]
        print(len(result),result)
        overall+=result
        print("\n")

    print(len(set(overall)),set(overall))

# X = data_processing.get_X(global_variables.numericas+global_variables.binarias+global_variables.categoricas+global_variables.ordinais)

# # Eliminate Low Variance Features

# print(X.shape)
# selector = VarianceThreshold(threshold=0.1)
# print(list(filter(lambda x: x[1] == False, list(zip(X.columns,selector.fit(X).get_support())))))
# X = selector.transform(X)
# print(X.shape)


# # Plot Correlation Matrix

# import seaborn as sns
# import matplotlib.pyplot as plt
# corr = X.corr()
# mask = np.triu(np.ones_like(corr, dtype=bool))
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()