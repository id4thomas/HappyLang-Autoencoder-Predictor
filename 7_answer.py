import csv
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def my_svm_s(kernel,un):
    #word_size=3
    #k=5
    print(kernel)
    if un==1:
        print("combined")
        my_data = genfromtxt('./picked/combined_d.csv', delimiter=',')
        my_label = genfromtxt('./picked/combined_l.csv', delimiter=',')
    else:
        print("original")
        my_data = genfromtxt('./processed/labeled_train_data_oh.csv', delimiter=',')
        my_label = genfromtxt('./processed/labeled_train_label_a.csv', delimiter=',')

    test_d=genfromtxt('./test/test_oh.csv', delimiter=',')
    #test_l=genfromtxt('./processed/labeled_test_label_a.csv', delimiter=',')
    #X=my_data[:,:-(2+len(concepts))]
    #X_id=my_data[:,0]
    X=my_data[:,:]
    y_social=my_label[:]
    #X=my_data[:,:999]
    #X=my_data[:,:-2]
    X_id=test_d[:,0]
    X_test=test_d[:,1:]
    #y_test=test_l[:]
    gamma=0.1
    clf = svm.SVC(gamma=gamma,kernel=kernel)
    clf.fit(X, y_social)
    y_pred=clf.predict(X_test)
    #print(precision_recall_fscore_support(y_test, y_pred, average=None))
    #print(accuracy_score(y_test, y_pred))
    return y_pred

def my_svm_a(kernel,un):
    #word_size=3
    #k=5
    print(kernel)
    if un==1:
        print("combined")
        my_data = genfromtxt('./picked/combined_d_a_balanced.csv', delimiter=',')
        my_label = genfromtxt('./picked/combined_l_a_balanced.csv', delimiter=',')
    else:
        print("original")
        my_data = genfromtxt('./processed/labeled_train_data_oh.csv', delimiter=',')
        my_label = genfromtxt('./processed/labeled_train_label_a.csv', delimiter=',')

    test_d=genfromtxt('./test/test_oh.csv', delimiter=',')
    #test_l=genfromtxt('./processed/labeled_test_label_a.csv', delimiter=',')
    #X=my_data[:,:-(2+len(concepts))]

    X=my_data[:,:]
    y_agency=my_label[:]
    #X=my_data[:,:999]
    #X=my_data[:,:-2]
    X_id=test_d[:,0]
    X_test=test_d[:,1:]
    #y_test=test_l[:]
    gamma=0.1
    clf = svm.SVC(gamma=gamma,kernel=kernel)
    clf.fit(X, y_agency)
    y_pred=clf.predict(X_test)
    #print(precision_recall_fscore_support(y_test, y_pred, average=None))
    #print(accuracy_score(y_test, y_pred))

    return y_pred
#my_svm('linear',0)
social_pred=my_svm_s('linear',1)#social
agency_pred=my_svm_s('rbf',1)

test_d=genfromtxt('./test/test_oh.csv', delimiter=',')

X_id=test_d[:,0]
to_write=[]
for i in range(len(X_id)):
    item=[]
    item.append(X_id[i])
    if agency_pred[i]==0:
        item.append('no')
    else:
        item.append('yes')
    if social_pred[i]==0:
        item.append('no')
    else:
        item.append('yes')
    #item.append(social_pred[i])
    to_write.append(item)

a_file=open("./test/test_answer_3.csv", 'w')
a_file.write("hmid,agency,social\n")
#input train test
wr = csv.writer(a_file, dialect='excel')
for x in to_write:
    wr.writerow(x)
a_file.close()

#my_svm('rbf',0)
#my_svm('rbf',1)
#my_knn(1,40)

#X_train, X_test, y_train, y_test = train_test_split(X, y_social, test_size=0.1)

#print("Social KNN")
#neigh = KNeighborsClassifier(n_neighbors=k)
#neigh.fit(X_train, y_train)
#y_pred=neigh.predict(X_test)
#print(neigh.score(X_test,y_test))
#cf=confusion_matrix(y_test, y_pred)
#print(cf)
#print(precision_recall_fscore_support(y_test, y_pred))


#X_train, X_test, y_train, y_test = train_test_split(X, y_agency, test_size=0.1)

#print("Agency KNN")
#neigh = KNeighborsClassifier(n_neighbors=k)
#neigh.fit(X_train, y_train)
#print(neigh.score(X_test,y_test))
#cf=confusion_matrix(y_test, y_pred)
#print(cf)
#print(precision_recall_fscore_support(y_test, y_pred))
