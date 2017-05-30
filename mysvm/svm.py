# File: svm.py
# Author: Indrajith Indraptrastham
# Date: Sun Apr 30 23:23:52 IST 2017


import numpy as np 
from . import acc
from . import feature
from sklearn import svm
from sklearn.model_selection import cross_val_score
import random
import itertools
import os
import pkg_resources
from sklearn.externals import joblib
#load pre saved variables
resource_package = __name__
resource_path = '/'.join(('', 'data/Xall.npy'))

Xall_path = pkg_resources.resource_filename(resource_package, resource_path)
Xall = np.load(Xall_path)
Yall = feature.geny(100)

resource_path = '/'.join(('', 'data/classifier_10class.pkl'))
clf_path = pkg_resources.resource_filename(resource_package, resource_path)
myclf = joblib.load(clf_path)

resource_path = '/'.join(('', 'data/classifier_10class_prob.pkl'))
clf_path = pkg_resources.resource_filename(resource_package, resource_path)
clfprob = joblib.load(clf_path)

label = feature.getlabels()
"""
def svms():
    #linear kernel=======================================================
    print('='*5+' Linear Kernel SVM '+'='*5)
    clf = svm.LinearSVC(C=10000)
    clf.fit(X90,Y90)

    #cross validation
    resub = clf.predict(X90)
    resubAcc = acc.get(resub,Y90)
    #test case
    print('Resubstituion Accuracy : ',resubAcc)
    resTest = clf.predict(xt10)
    tesAcc = acc.get(resTest,yt10)
    print('Test case accuracy : ',tesAcc)
    scores = cross_val_score(clf, X90, Y90, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('='*30)

    #Polynomial kernel=======================================================
    print('='*5+' Polynomial Kernel SVM '+'='*5)
    clf = svm.SVC(kernel='poly',C=1000)
    clf.fit(X,Y)

    #cross validation
    resub = clf.predict(X)
    resubAcc = acc.get(resub,Y)
    #test case
    print('Resubstituion Accuracy : ',resubAcc)
    resTest = clf.predict(xt10)
    tesAcc = acc.get(resTest,yt10)
    print('Test case accuracy : ',tesAcc)

    scores = cross_val_score(clf, xt, yt, cv=5)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('='*30)

"""

def poly(X,Y):
    #Polynomial kernel=======================================================
    clf = svm.SVC(kernel='poly',C=1)
    clf.fit(X,Y)
    #cross validation
    #resub = clf.predict(X)
    #resubAcc = acc.get(resub,Y)
    #test case
    #resTest = clf.predict(xt)
    #tesAcc = acc.get(resTest,yt)
    #scores = cross_val_score(clf, X, Y, cv=5)

    #return scores.mean()
    return clf


def fit(train_percentage,fold=5):
    """ Accepts parameter: 
            train_percentage;
            fold;
    """
    #creates a matrix of size 10x100x104
    #return clf
    resTrain =0
    resTest = 0
    score = 0
    scores = 0

    for folds in range(fold):
        #init
        flag = True
        flag_train = True
        start = 0
        train_matrix = np.array([])
        test_matrix = np.array([])
        Xindex = []
        Tindex = []

        for class_counter in range(10):
            stack = list(range(start, start+100))  #create an index of size 100
            for song_counter in range( int(train_percentage) ):
                index = random.choice(stack)      #randomly choose numbers from index
                stack.remove(index)               #remove the choosen number from index
                random_song = Xall[index]         #select songs from that index for training
                Xindex.append(index)
                if flag:
                    train_matrix = random_song
                    flag = False
                else:
                    train_matrix = np.vstack([train_matrix, random_song])
            start += 100

            #select the remaning songs from the stack for testing
            for test_counter in range(100 - train_percentage):
                Tindex.append(stack[test_counter])
                if flag_train:
                    test_matrix = Xall[stack[test_counter]]
                    flag_train = False
                else:
                    test_matrix = np.vstack([test_matrix, Xall[stack[test_counter]]])

        Y = feature.geny(train_percentage) 
        y = feature.geny(100 - train_percentage)
        #training accuracy
        clf = svm.SVC(kernel='poly',C=1,probability=True)
        clf.fit(train_matrix, Y)
        return clf


def getprob(filename):
    x = feature.extract(filename)
    clf = clfprob
    prob = clf.predict_proba(x)[0]
    dd = dict(zip(feature.getlabels(),prob))
    # max probablity 
    m = max(dd,key=dd.get)
    print(m, dd[m])
    
    """ to print all probablity and difference from max
    for i in dd:
        if i != m:
            if dd[m] - dd[i] :
                print(i, dd[i], '| diff:',dd[m] - dd[i])
    """

    sorted_genre = sorted(dd,key=dd.get,reverse=True)
    has_features_of = []
    for i in sorted_genre:
        if dd[i] > 0.15:
            has_features_of.append(i)


    return dd, has_features_of
    



def random_cross_validation(train_percentage,fold):
    """ Accepts parameter: 
            train_percentage;
            fold;
    """
    #creates a matrix of size 10x100x104
    resTrain =0
    resTest = 0
    score = 0
    scores = 0

    for folds in range(fold):
        #init
        flag = True
        flag_train = True
        start = 0
        train_matrix = np.array([])
        test_matrix = np.array([])
        Xindex = []
        Tindex = []

        for class_counter in range(10):
            stack = list(range(start, start+100))  #create an index of size 100
            for song_counter in range( int(train_percentage) ):
                index = random.choice(stack)      #randomly choose numbers from index
                stack.remove(index)               #remove the choosen number from index
                random_song = Xall[index]         #select songs from that index for training
                Xindex.append(index)
                if flag:
                    train_matrix = random_song
                    flag = False
                else:
                    train_matrix = np.vstack([train_matrix, random_song])
            start += 100

            #select the remaning songs from the stack for testing
            for test_counter in range(100 - train_percentage):
                Tindex.append(stack[test_counter])
                if flag_train:
                    test_matrix = Xall[stack[test_counter]]
                    flag_train = False
                else:
                    test_matrix = np.vstack([test_matrix, Xall[stack[test_counter]]])

        #print(train_matrix.shape)
        #print(test_matrix.shape)

        Y = feature.geny(train_percentage) 
        y = feature.geny(100 - train_percentage)
        #training accuracy
        clf = svm.SVC(kernel='poly',C=1)
        clf.fit(train_matrix, Y)
        #train case
        #scores = cross_val_score(clf, train_matrix, feature.geny(train_percentage), cv=5)
        #print("acc train",scores.mean())
        #test case
        #scores = cross_val_score(clf, test_matrix, feature.geny(100 - train_percentage), cv=5)
        #print("acc test",scores.mean())
        res = clf.predict(train_matrix)
        #print(acc.get(res,Y))
        resTrain += acc.get(res,Y)
        res = clf.predict(test_matrix)
        resTest += acc.get(res,y)
        #print(acc.get(res,y))
        #return test_matrix, train_matrix
        #scores = cross_val_score(clf, test_matrix, y, cv=5)
        #print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
        #score = 100 *clf.score(test_matrix,y)
        #print(score)
        #scores += score

    #print(scores/fold)
    #print(resTrain)
    #print(resTest)
    print("Training accuracy with %d fold %f: " % (int(fold), resTrain / int(fold)))
    print("Testing accuracy with %d fold %f: " % (int(fold), resTest / int(fold)))
    


def findsubclass(class_count):
    """ 
    Finds the combination of classes NCR
    """
    class_l = list(range(10))
    flag = True
    labels = np.array([])
    for i in itertools.combinations(class_l,class_count):
        if flag:
            labels = i
            flag = False
        else:
            labels = np.vstack([labels, i])
    return labels

def gen_sub_data(class_l):
    all_x = np.array([])
    flag = True;

    for class_index in class_l:
        if class_index != 0:
            class_index *= 100
        if flag:
            all_x = Xall[ class_index : class_index + 100 ]
            flag = False
        else: 
            all_x = np.vstack([all_x, Xall[ class_index : class_index + 100 ]])
    
    return all_x


def fitsvm(Xall,Yall,class_l,train_percentage,fold):
    """ Accepts parameter: 
            train_percentage;
            fold;
    """
    #creates a matrix of size 10x100x104
    resTrain =0
    resTest = 0
    score = 0
    scores = 0

    for folds in range(fold):
        #init
        flag = True
        flag_train = True
        start = 0
        train_matrix = np.array([])
        test_matrix = np.array([])
        Xindex = []
        Tindex = []

        for class_counter in range(class_l):
            stack = list(range(start, start+100))  #create an index of size 100
            for song_counter in range( int(train_percentage) ):
                index = random.choice(stack)      #randomly choose numbers from index
                stack.remove(index)               #remove the choosen number from index
                random_song = Xall[index]         #select songs from that index for training
                Xindex.append(index)
                if flag:
                    train_matrix = random_song
                    flag = False
                else:
                    train_matrix = np.vstack([train_matrix, random_song])
            start += 100

            #select the remaning songs from the stack for testing
            for test_counter in range(100 - train_percentage):
                Tindex.append(stack[test_counter])
                if flag_train:
                    test_matrix = Xall[stack[test_counter]]
                    flag_train = False
                else:
                    test_matrix = np.vstack([test_matrix, Xall[stack[test_counter]]])
        Y = feature.gen_suby(class_l, train_percentage) 
        y = feature.gen_suby(class_l, 100 - train_percentage)
        #training accuracy
        clf = svm.SVC(kernel='poly',C=1)
        clf.fit(train_matrix, Y)
        #train case
        res = clf.predict(train_matrix)
        #print(acc.get(res,Y))
        resTrain += acc.get(res,Y)
        res = clf.predict(test_matrix)
        resTest += acc.get(res,y)

    return clf , resTest / int(fold)

def best_combinations(class_l, train_percentage, fold):
    class_comb = findsubclass(class_l)
    avg = []
    X = gen_sub_data(class_comb[0])
    Y = feature.gen_suby(class_l,100)
    for class_count in range(class_comb.shape[0]):
        all_x = gen_sub_data( class_comb[ class_count ] )
        all_y = feature.gen_suby(class_l,100)
        clf , score = fitsvm(all_x, all_y, class_l, train_percentage, fold)
        avg.append(score)
        print(score)
        print(class_count)
    maxAvg = max(avg)
    maxIndex = [i for i, j in enumerate(avg) if j >= (maxAvg - 2)]
    print("Maximum accuracy:",maxAvg)
    print("Best combinations:")
    for i in maxIndex:
        print(class_comb[i])
    return  avg


def getgenre(filename):

    music_feature =  feature.extract(os.path.abspath(os.path.dirname(__name__)) \
        +'/django-jquery-file-upload/' +filename)
    clf = myclf
    return clf.predict(music_feature)

