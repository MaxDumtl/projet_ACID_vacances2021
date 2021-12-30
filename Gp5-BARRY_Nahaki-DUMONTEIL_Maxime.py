from numpy.matrixlib import defmatrix
from pandas.core import strings
from sklearn import datasets
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical



import numpy as np
from numpy.linalg import eig 
from numpy.linalg import norm
from numpy.polynomial import Polynomial

import scipy
from scipy.stats import norm
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from time import time

import pandas as pa

from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from random import sample, choices
import random as rd
from collections import Counter
import warnings


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# ============================TEST IMPORT LIB ==================================
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

# print("Shape X_train = ", X_train.shape)
# print("Shape y_train = ", y_train.shape)
# print("Shape X_test = ", X_test.shape)
# print("Shape X_test = ", y_test.shape)

# y = np.array([1,2,3])
# print(y)

# data = pa.DataFrame()
# print(data)

# ------------------------------------------------

# print(y_train[0])
# plt.imshow(X_train[0], cmap=cm.Greys)
# plt.show()
# ================================================================================


# =========================================STRUCTURE DATA=========================================
dict = {'Image' : [i for i in range(X_train.shape[0])],
        'Type' : y_train}
data_train = pa.DataFrame(data=dict)

# liste de vecteur pour chaque image avec leur types de vêtement en dernière élément
X_train_1D = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test_1D = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

data_knn = [np.append(X_train_1D[i][:X_train_1D.shape[1]], y_train[i]) for i in range(len(y_train))]
# for i in range(X_train_1D.shape[1]):
#     data_train['Pixel'+ str(i)] = [X_train_1D[j][i] for j in range(X_train_1D.shape[0])]
# pa.concat()
# print(data_train.head())

# classification sklearn
# X = data_train[['Image', 'Type']]
# Y = data_train['Type']
#=================================================================================================

# print(X_train_1D.shape)
# print(y_train.shape)

#==========================================ACP-SKLEARN==========================================

# scaler = StandardScaler()
# z = scaler.fit_transform(X_train_1D)

# nb_axes_1 = 10
# nb_axes_2 = 50
# nb_axes_3 = 100
# nb_axes_4 = 300

# acp_1 = PCA(n_components=nb_axes_1)
# acp_2 = PCA(n_components=nb_axes_2)
# acp_3 = PCA(n_components=nb_axes_3)
# acp_4 = PCA(n_components=nb_axes_4)
# print(acp)
# coord_1 = acp_1.fit_transform(z)
# coord_2 = acp_2.fit_transform(z)
# coord_3 = acp_3.fit_transform(z)
# coord_4 = acp_4.fit_transform(z)
# print(acp.n_components_)

# plt.grid()
# plt.plot(np.arange(1,acp.n_components_+1),acp.explained_variance_ratio_) 
# plt.title("Scree plot") 
# plt.ylabel("Eigen values") 
# plt.xlabel("Factor number") 
# plt.show()

# print(acp_1.explained_variance_ratio_.sum())
# print(acp_2.explained_variance_ratio_.sum())
# print(acp_3.explained_variance_ratio_.sum())
# print(acp_4.explained_variance_ratio_.sum())

# *******************************************************************
# ******************pour une variance de plus 0.9 = 139**************
# *******************************************************************
# p = 100
# total_variance = 0
# while(total_variance < .9) :
#     print(p)
#     acp = PCA(n_components=p)
#     coord = acp.fit_transform(z)
#     total_variance = acp.explained_variance_ratio_.sum()
#     p += 1
    
# print('p = ', p)
# print('total explained variance ratio: ', total_variance)
#*******************************************************************

def trainACP(classifier, axes, z_train, z_test):
    acp = PCA(n_components=axes)

    Xp_train_1D = acp.fit_transform(z_train)
    Xp_test_1D = acp.fit_transform(z_test)

    classifier.fit(Xp_train_1D, y_train)

    y_pred = classifier.predict(Xp_test_1D)
    acc = accuracy_score(y_test, y_pred)
    
    return acc

#Préparation des données pour une réduction ACP
scaler_b = StandardScaler()
z_train = scaler_b.fit_transform(X_train_1D)
z_test = scaler_b.fit_transform(X_test_1D)


#==========================================KNN-SKLEARN==========================================
# OBSERVATION :
# Accuracy du modèle = 85,54%

knn = KNeighborsClassifier()

t = time() 
knn.fit(X_train_1D, y_train)
t = time() - t

y_pred = knn.predict(X_test_1D)

# cm_knn = confusion_matrix(y_test, y_pred)
# print(cm_knn)
acc_knn = accuracy_score(y_test, y_pred)
print("Accuracy for knn = ", '{:.2%}'.format(acc_knn), " in time = ", t)


#=======================================KNN_ACP-SKLEARN==========================================
# OBSERVATION :
# L'ACP n'améliore pas l'accurasy du modèle pour un classifieur knn au contraire il la dégrade, le temps d'entrainement est également 
# décuplé
# *************************************************************************
# -> la meilleur accurcacy que j'ai pu trouver était de 80,41% pour 11 axes 
# *************************************************************************
# lorsque l'on réduit les dimensions à partir du nombre d'axes optimale (11) l'accuracy diminue également
# lorsque l'on augmente les dimensions à partir du nombre d'axes optimale (11) l'accuracy diminue également

knn = KNeighborsClassifier()
p_b = 11

t = time() 
acc_knn_acp = trainACP(knn, p_b, z_train, z_test)
t = time() - t

print("Best accuracy for knn (ACP - ", p_b, " axes) = ",'{:.2%}'.format(acc_knn_acp), " in time = ", t)

# --------------------------------------------------TEST--------------------------------------------------------------
# max_acc = trainACP(knn, 1, z_train, z_test)
# best_p = 1
# p_b = best_p + 1
# acc = trainACP(knn, p_b, z_train, z_test)
# while(p_b < 784):
#     if(acc > best_p):
#         max_acc = acc
#         best_p = p_b

#     p_b += 1
#     acc = trainACP(knn, p_b, z_train, z_test)
#     print("Accuracy for knn (ACP - ", p_b, " axes) = ",'{:.2%}'.format(acc))

# print("Best accuracy for bayes (ACP - ", best_p, " axes) = ",'{:.2%}'.format(max_acc), " in time = ", t)
# --------------------------------------------------------------------------------------------------------------------


#=========================================BAYES-SKLEARN=========================================
# OBSERVATION :
# Accuracy du modèle = 58,56%

nb = GaussianNB()

t = time() 
nb.fit(X_train_1D, y_train)
t = time() - t

y_pred = nb.predict(X_test_1D)

acc_b = accuracy_score(y_test, y_pred)
print("Accuracy for bayes = ", '{:.2%}'.format(acc_b), " in time = ", t)

#======================================BAYES_ACP-SKLEARN=========================================

# OBSERVATION :
# plus le nombre axes est petit meilleur est l'accuracy du modèle et le temps de d'entrainement est réduit
# *************************************************************************
# -> le meilleur accurcacy que j'ai pu trouver était de 69,10% pour 11 axes 
# *************************************************************************

nb = GaussianNB()
p_b = 11

t = time() 
acc_b_acp = trainACP(nb, p_b, z_train, z_test)
t = time() - t

print("Best accuracy for bayes (ACP - ", p_b, " axes) = ",'{:.2%}'.format(acc_b_acp), " in time = ", t)

# --------------------------------------------------TEST--------------------------------------------------------------
# max_acc = trainACP(nb, 1, z_train, z_test)
# best_p = 1
# p_b = best_p + 1
# acc = trainACP(nb, p_b, z_train, z_test)
# while(p_b < 784):
#     if(acc > best_p):
#         max_acc = acc
#         best_p = p_b

#     p_b += 1
#     acc = trainACP(nb, p_b, z_train, z_test)
#     print("Accuracy for bayes (ACP - ", p_b, " axes) = ",'{:.2%}'.format(acc))

# print("Best accuracy for bayes (ACP - ", best_p, " axes) = ",'{:.2%}'.format(max_acc), " in time = ", t)
# --------------------------------------------------------------------------------------------------------------------


#------------------------------------------------BASE DE TRAVAIL------------------------------------------------------
# nb = GaussianNB()
# scaler_b = StandardScaler()

# z_train = scaler_b.fit_transform(X_train_1D)
# z_test = scaler_b.fit_transform(X_test_1D)
# p_b = 11
# acp = PCA(n_components=p_b)
# Xp_train_1D = acp.fit_transform(z_train)
# Xp_test_1D = acp.fit_transform(z_test)

# t = time() 
# nb.fit(Xp_train_1D, y_train)
# t = time() - t

# y_pred = nb.predict(Xp_test_1D)

# cm_bayes = confusion_matrix(y_test, y_pred)
# print(cm_bayes)
# acc_bayes_acp = accuracy_score(y_test, y_pred)
# --------------------------------------------------------------------------------------------------------------------





#=========================================KNN-NEIGNBOOR=========================================
def euclidean(u, v):
    return np.sqrt(np.sum((u -v[:-1])**2))

def distances(u, dataset):
    dist = []
    for d in dataset:
        dist.append(euclidean(u, d))
    return dist

def voisins(u, datatset, k):
    distances = []

    for d in datatset:
        distances.append((d, euclidean(u, d)))
    #distances.sort(key=lambda tup: tup[1])
    distances.sort(key=lambda tup: tup[1])
    neighs = []
    for i in range(k):
        neighs.append(distances[i][0])
    #print('u: ', u, ' its neighbors: ', neighs)
    return neighs

  

# print(classifier(X_train_1D[0], X_train_1D[:3], 1))

# print("\n", X_train_1D[0])
# print(X_train_1D[:][0:2].shape)
# print(len(X_train_1D[:][0:2]))
# print(X_train_1D[:][0:2][0])
# print(X_train_1D[:][0:2][1])



#=================================================================================================

