import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from os.path import join as opj
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import pylab


train = pd.read_json("/home/sushi/Desktop/Data/train.json")
#print(train)

#replace 'na'and drop those values for inc_angle
#print(train['inc_angle'])
train.inc_angle.replace({'na':np.nan},inplace=True)
train.drop(train[train['inc_angle'].isnull()].index,inplace=True)

#create numpy matrix from dataset
X_train_HH=np.array([np.array(band).astype(np.float32) for band in train.band_1])
X_train_HV=np.array([np.array(band).astype(np.float32) for band in train.band_2])
r, _ = X_train_HH.shape
#X_train_angle=np.array([np.array(angle).astype(np.float32) for angle in train.inc_angle]).T.reshape(r,1)
y_train=train.is_iceberg.values.astype(np.float32)

print(X_train_HH.shape, X_train_HV.shape,(X_train_HH +X_train_HV)/2)
X_train=np.concatenate((X_train_HH, X_train_HV,((X_train_HH +X_train_HV)/2)), axis=1)

X_new_train, X_test, y_new_train, y_test = train_test_split(X_train, y_train, test_size=0.75)

print(X_new_train.shape)
print(X_test.shape)

#scale each attribute
scaler=MaxAbsScaler()
X_new_train_maxabs=scaler.fit_transform(X_new_train)
X_test_maxabs=scaler.fit_transform(X_test)

#tuning the parameters
#create the SVM instance using Radial Basis Function (rbf) kernel
clf = svm.SVC(kernel='rbf', probability=True)


#set the range of hyper-parameter to tune the SVM classifier
C_range= [ 1,5, 10, 50, 100]
gamma_range = [ 0.00001,0.0001, 0.001,0.01,0.1]
param_grid_SVM = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(clf, param_grid=param_grid_SVM, cv=3, scoring='roc_auc')

grid.fit(X_new_train_maxabs,y_new_train)
pred_y_test = grid.predict (X_test_maxabs)
accuracy_test=accuracy_score(y_test,pred_y_test)




#Accuracy score on the test dataset

print("The best parameters are %s with a score of %0.2f"% (grid.best_params_,grid.best_score_))
print (accuracy_test)

