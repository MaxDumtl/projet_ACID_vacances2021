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

from sklearn import tree
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

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

# print(y_train[0])
# plt.imshow(X_train[0], cmap=cm.Greys)
# plt.show()
# ================================================================================


# =========================================STRUCTURE DATA=========================================
dict = {'Image' : [i for i in range(X_train.shape[0])],
        'Type' : y_train}
data_train = pa.DataFrame(data=dict)

# liste de vecteur pour chaque image avec leur types de v??tement en derni??re ??l??ment
X_train_1D = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test_1D = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

data_knn = [np.append(X_train_1D[i][:X_train_1D.shape[1]], y_train[i]) for i in range(len(y_train))]
# for i in range(X_train_1D.shape[1]):
#     data_train['Pixel'+ str(i)] = [X_train_1D[j][i] for j in range(X_train_1D.shape[0])]
# pa.concat()
# print(data_train.head())
#=================================================================================================


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
# print(acp_1)
# print(acp_2)
# print(acp_3)
# print(acp_4)
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

#Pr??paration des donn??es pour une r??duction ACP
scaler_b = StandardScaler()
z_train = scaler_b.fit_transform(X_train_1D)
z_test = scaler_b.fit_transform(X_test_1D)


#==========================================LDA-SKLEARN==========================================

# Pr??paration des donn??es pour une r??duction LDA
lda_train = LinearDiscriminantAnalysis()
lda_test = LinearDiscriminantAnalysis()
X_train_lda = lda_train.fit_transform(X_train_1D, y_train)
X_test_lda = lda_test.fit_transform(X_test_1D, y_test)

# print(lda_train.explained_variance_ratio_)
# print(lda_train.explained_variance_ratio_)

# print(lda_test.explained_variance_ratio_)

# plt.scatter(X_train_lda[:,0], np.zeros(X_train_lda.shape[0]), c=y_train)
# plt.show()

# plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train)
# plt.xlabel('LDA1')
# plt.ylabel('LDA2')
# plt.show()




#==========================================KNN-SKLEARN==========================================
# OBSERVATION :
# Accuracy du mod??le = 85,54%
# Entra??nement du mod??le = 0.0050 / 0.0089


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
# L'ACP n'am??liore pas l'accuracy du mod??le pour un classifieur "knn" au contraire il la d??grade (-5%), le temps d'entrainement  
# est ??galement d??cupl??
# *************************************************************************
# -> la meilleur accurcacy que j'ai pu trouver ??tait de 80,41% / 80.37% pour 11 axes 
# -> le temps d'entra??nement associ?? est de 3.7038
# *************************************************************************
# lorsque l'on r??duit les dimensions ?? partir du nombre d'axes optimale (11) l'accuracy diminue ??galement
# lorsque l'on augmente les dimensions ?? partir du nombre d'axes optimale (11) l'accuracy diminue ??galement


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

#=======================================KNN_LDA-SKLEARN==========================================
# OBSERVATION :
# Accuracy du mod??le = 59,78%
# Entra??nement du mod??le = 0.1794 / 0.1415 / 0.1070 
#  - La r??duction de dimension par LDA d??grade (-25%) l'accuracy du mod??le avec une classification par "knn" et augmente l??g??rement
#    le temps d'entra??nement du mod??le  
#  - La r??duction par LDA est (nettement -20%) moins efficace que la r??duction par ACP pour la classification du corpus 


knn = KNeighborsClassifier()

t = time() 
knn.fit(X_train_lda, y_train)
t = time() - t

y_pred = knn.predict(X_test_lda)
acc_knn_lda = accuracy_score(y_test, y_pred)

print("Accuracy for knn (LDA) =  ",'{:.2%}'.format(acc_knn_lda), " in time = ", t)

#=========================================BAYES-SKLEARN=========================================
# OBSERVATION :
# Accuracy du mod??le = 58,56%
# Entra??nement du mod??le = 0.6572 / 0.5744


nb = GaussianNB()

t = time() 
nb.fit(X_train_1D, y_train)
t = time() - t

y_pred = nb.predict(X_test_1D)

acc_b = accuracy_score(y_test, y_pred)
print("Accuracy for bayes = ", '{:.2%}'.format(acc_b), " in time = ", t)

#======================================BAYES_ACP-SKLEARN=========================================
# OBSERVATION :
# plus le nombre axes est petit meilleur est l'accuracy du mod??le
# *************************************************************************
# -> la meilleur accurcacy que j'ai pu trouver ??tait de 69,10% / 69.12% pour 11 axes 
# -> le temps d'entra??nement correspondant est de 2.727
# *************************************************************************
# comme pour la r??duction avec ACP pour la classification par knn avec bayes on observe ??galement que 
#  - lorsque l'on r??duit les dimensions ?? partir du nombre d'axes optimale (11) l'accuracy diminue ??galement
#  - lorsque l'on augmente les dimensions ?? partir du nombre d'axes optimale (11) l'accuracy diminue ??galement

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

#======================================BAYES_LDA-SKLEARN=========================================
# OBSERVATION :
# Accuracy du mod??le = 60.97%
# Entra??nement du mod??le = 0.0228 / 0.0129
#  - La r??duction de dimension par LDA permet d'am??liorer l'accuracy du mod??le avec une classification par Bayes et r??duit ??galement 
#    le temps d'entra??nement du mod??le 
#  - La r??duction par LDA est moins efficace que la r??duction par ACP pour la classification du corpus 


nb = GaussianNB()

t = time() 
nb.fit(X_train_lda, y_train)
t = time() - t

y_pred = nb.predict(X_test_lda)
acc_b = accuracy_score(y_test, y_pred)
print("Accuracy for bayes (LDA) = ", '{:.2%}'.format(acc_b), " in time = ", t)


"""
Observartion g??n??rale sur les classifications par knn et bayes coupl??es aux techniques de r??duction ACP et LDA

Le meilleur classifieur pour notre corpus ?? ce niveau est le knn 
Pour ces deux premiers classifieurs la r??duction par ACP augmente (nettement) le temps d'entrainement du mod??le

-> Sans r??duction
Le meilleur des deux classifieurs pour notre corpus est le knn (85.54% > 58.56%), le temps d'entra??nement du mod??le est ??galement meilleur (0.0089 < 0.5744)

-> Avec r??duction
(KNN)
 - La r??duction de dimension d??grade la classification par knn (LDA plus que l'ACP) le temps d'entra??nement du mod??le est ??galement 
   d??grad?? (ACP beaucoup plus que la LDA 3.7038 > 0.1070 > 0.0089)

A L'INVERSE

(BAYES)
 - La r??duction de dimension am??liore la classification par bayes (ACP plus que la LDA) le temps d'entra??nement du mod??le est n??anmoins 
   augment?? pour une r??duction par ACP (2.7279 > 0.5744) mais il est r??duit avec une r??duction par LDA (0.0129 < 0.5744)
"""

#======================================DECISION_TREE-SKLEARN=========================================
# OBSERVATION :
# Accuracy du mod??le = 78.99% / 79.19%
# Entra??nement du mod??le = 39.2504 / 51.4929

dt = tree.DecisionTreeClassifier()

t = time() 
dt.fit(X_train_1D, y_train)
t = time() - t

y_pred = dt.predict(X_test_1D)
acc_tree = metrics.accuracy_score(y_test, y_pred)
print('Accuracy for decision tree: ','{:2.2%}'.format(acc_tree), " in time = ", t)

#===================================DECISION_TREE_ACP-SKLEARN========================================
# OBSERVATION :
# 11 - 71.80% -3.9518
# La r??duction ACP am??liore nettement le temps d'entra??nement (de 36 sec) du mod??le mais r??duit (de 7%) l'accuracy de la classification

dt = tree.DecisionTreeClassifier()
p_b = 11

t = time() 
acc_tree_acp = trainACP(dt, p_b, z_train, z_test)
t = time() - t

print("Best accuracy for decision tree (ACP - ", p_b, " axes) = ",'{:.2%}'.format(acc_tree_acp), " in time = ", t)

# --------------------------------------------------TEST--------------------------------------------------------------
# max_acc = trainACP(dt, 1, z_train, z_test)
# best_p = 1
# p_b = best_p + 1
# acc = trainACP(dt, p_b, z_train, z_test)
# while(p_b < 784):
#     if(acc > best_p):
#         max_acc = acc
#         best_p = p_b

#     p_b += 1
#     acc = trainACP(dt, p_b, z_train, z_test)
#     print("Accuracy for decision tree (ACP - ", p_b, " axes) = ",'{:.2%}'.format(acc))

# print("Best accuracy for decision tree (ACP - ", best_p, " axes) = ",'{:.2%}'.format(max_acc), " in time = ", t)
# --------------------------------------------------------------------------------------------------------------------

#===================================DECISION_TREE_LDA-SKLEARN=======================================
# OBSERVATION :
# Accuracy du mod??le = 52.93% / 51.56%
# Entra??nement du mod??le = 0.9530 /  0.9684
# La r??duction LDA am??liore nettement le temps d'entra??nement du mod??le (le fait passer ?? moins d'une seconde) mais r??duit (de 19%) 
# l'accuracy de la classification



dt = tree.DecisionTreeClassifier()

t = time() 
dt.fit(X_train_lda, y_train)
t = time() - t

y_pred = dt.predict(X_test_lda)
acc_tree_lda = accuracy_score(y_test, y_pred)
print("Accuracy for decision tree (LDA) = ", '{:.2%}'.format(acc_tree_lda), " in time = ", t)

#======================================RANDOM_FOREST-SKLEARN=========================================
# OBSERVATION :
# Accuracy du mod??le = 87.65%
# Entra??nement du mod??le = 79.7895


rf = RandomForestClassifier()

t = time() 
rf.fit(X_train_1D, y_train)
t = time() - t

y_pred = rf.predict(X_test_1D)
acc_rd_forest = metrics.accuracy_score(y_test, y_pred)
print('Accuracy for random forest: ','{:2.2%}'.format(acc_rd_forest), " in time = ", t)

#===================================RANDOM_FOREST_ACP-SKLEARN========================================
# OBSERVATION :
# 11 - 81.40% - 23.0976
# La r??duction ACP am??liore nettement le temps d'entra??nement du mod??le (reduit de 56 sec) mais r??duit (de 6%) l'accuracy de 
# la classification

rf = RandomForestClassifier()
p_b = 11

t = time() 
acc_forest_acp = trainACP(rf, p_b, z_train, z_test)
t = time() - t

print("Best accuracy for random forest (ACP - ", p_b, " axes) = ",'{:.2%}'.format(acc_forest_acp), " in time = ", t)

# --------------------------------------------------TEST--------------------------------------------------------------
# max_acc = trainACP(rf, 1, z_train, z_test)
# best_p = 1
# p_b = best_p + 1
# acc = trainACP(rf, p_b, z_train, z_test)
# while(p_b < 784):
#     if(acc > best_p):
#         max_acc = acc
#         best_p = p_b

#     p_b += 1
#     acc = trainACP(rf, p_b, z_train, z_test)
#     print("Accuracy for random forest (ACP - ", p_b, " axes) = ",'{:.2%}'.format(acc))

# print("Best accuracy for random forest (ACP - ", best_p, " axes) = ",'{:.2%}'.format(max_acc), " in time = ", t)
# --------------------------------------------------------------------------------------------------------------------

#===================================RANDOM_FOREST_LDA-SKLEARN=======================================
# OBSERVATION :
# Accuracy du mod??le = 58.70% / 58.38%
# Entra??nement du mod??le = 45.8462 / 21.5101
# La r??duction LDA am??liore le temps d'entra??nement du mod??le (r??duit de 34 sec) mais r??duit (de 23%) l'accuracy de la classification
# La r??duction par ACP est meilleur en tout point par rapport ?? une r??duction par LDA (pour une classification par for??t al??atoire) 
# m??me si les deux r??ductions am??liores le temps d'entra??nement du mod??le les deux r??duisent l'accuracy de la classification (la LDA plus
# que l'ACP)

rf = RandomForestClassifier()

t = time() 
rf.fit(X_train_lda, y_train)
t = time() - t

y_pred = rf.predict(X_test_lda)
acc_forest_lda = accuracy_score(y_test, y_pred)
print("Accuracy for random forest (LDA) = ", '{:.2%}'.format(acc_forest_lda), " in time = ", t)



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



'''
Observations g??n??rale finale sur l'ensemble des classifieurs (knn, bayes, decision_tree random_forest):

De mani??re g??n??rale, le meilleur classifieur est la for??t al??atoire (environ 87% de qualit??)
pour classer les donn??es de ce corpus.

Dans la majorit?? des cas, les techniques de r??duction de dimensions ont pour effet de d??grader
la qualit?? du classifieur pour ce corpus. Il n'y a que pour le classifieur Bayes que les techniques
ACP et LDA ont donn?? de meilleurs r??sultats que le m??me classifieur sans r??duction.
Ces techniques ont ralenti l'entrainement et ont augment?? la complexit?? des mod??les KNN et Bayes sur
les donn??es mais ont acc??l??r?? le processus pour l'arbre de d??cision et la for??t al??atoire au d??triment des
r??sultats de leur classifications.

Sans r??duction, le meilleur classifieur est celui utilisant le mod??le de la for??t al??atoire tandis que
le pire semble ??tre Bayes (environ 58% de qualit??). 
Avec r??duction, le meilleur classifieur est encore la for??t al??atoire avec la technique ACP (environ 81%).

De mani??re g??n??rale, c'est la technique de r??ductions de l'ACP qui donne de meilleurs r??sultats sur ce
corpus de donn??es compar??s ?? LDA.
'''

