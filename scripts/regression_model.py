#!/usr/bin/python
import sys
import os
import csv
import time
import datetime
from types import *

import random

import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn_pandas import DataFrameMapper, cross_val_score

from mpl_toolkits.mplot3d import Axes3D

from patsy import dmatrices

import pymysql as mdb



# In[178]:

def regressionModel(df_sig_train, df_bkg_train, df_sig_test, df_bkg_test):

    # Reminder:
    # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #           intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

    #df_sig_train['X1X2'] = df_sig_train['PTS']*df_sig_train['AST']
    #df_sig_train['X1X1'] = df_sig_train['PTS']*df_sig_train['PTS']
    #df_sig_train['X2X2'] = df_sig_train['AST']*df_sig_train['AST']

    #df_bkg_train['X1X2'] = df_bkg_train['PTS']*df_bkg_train['AST']
    #df_bkg_train['X1X1'] = df_bkg_train['PTS']*df_bkg_train['PTS']
    #df_bkg_train['X2X2'] = df_bkg_train['AST']*df_bkg_train['AST']

    X_sig = np.array(df_sig_train)
    print 'X_sig.shape: ', X_sig.shape

    y_sig = np.array(X_sig.shape[0] * [1])
    print 'y_sig.shape: ', y_sig.shape

    X_bkg = np.array(df_bkg_train)
    print 'X_bkg.shape: ', X_bkg.shape

    y_bkg = np.array(X_bkg.shape[0] * [0])
    print 'y_bkg.shape: ', y_bkg.shape


    X = np.concatenate((X_sig, X_bkg))
    y = np.concatenate((y_sig, y_bkg))

    #print X
    print 'X.shape: ', X.shape
    print 'y.shape: ', y.shape

    # first way of doing preprocessing
    #X_scaled = preprocessing.scale(X)

    # second way of doing preprocessing
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    #print X_scaled, X_scaled.mean(axis=0), X_scaled.std(axis=0)


    model = LogisticRegression()

    model.fit(X, y)
    #model.fit(X_scaled, y)


    ###########################################################

    # 1. First, check the accuracy on the TRAINING set
    print 'model.score(X, y): ', model.score(X, y)
    #print 'model.score(X, y): ', model.score(X_scaled, y) 


    # 2. See what we get out of the box of TRAINING set without prediction (e.g. everytime predict 0)
    print 'y.mean(): ', y.mean()


    # 3A. Examine the coefficients
    #print "Coefficients: ", pd.DataFrame(zip(X, np.transpose(model.coef_)))


    # 3B. Calculating Error
    #predicted_train = model.predict(X)
    #print predicted_train
    #predicted_train_clever = (predicted_train + y).tolist()    
    #error = float(predicted_train_clever.count(1)) / float(len(predicted_train_clever))
    #print "Error: ", error_train


    # 4. Cross-validation

    #scores = cross_val_score(LogisticRegression(), X , y, 'accuracy', 4)
    #print "Cross-validation: ", scores
    #print "Cross-validation mean: ", scores.mean()


    # 5. Testing model performance using testing dataset
    #df_sig_test['X1X2'] = df_sig_test['PTS']*df_sig_test['AST']
    #df_sig_test['X1X1'] = df_sig_test['PTS']*df_sig_test['PTS']
    #df_sig_test['X2X2'] = df_sig_test['AST']*df_sig_test['AST']

    #df_bkg_test['X1X2'] = df_bkg_test['PTS']*df_bkg_test['AST']
    #df_bkg_test['X1X1'] = df_bkg_test['PTS']*df_bkg_test['PTS']
    #df_bkg_test['X2X2'] = df_bkg_test['AST']*df_bkg_test['AST']

    
    
    X_sig_test = np.array(df_sig_test)
    print 'X_sig_test.shape: ', X_sig_test.shape

    y_sig_test = np.array(X_sig_test.shape[0] * [1])
    print 'y_sig_test.shape: ', y_sig_test.shape

    X_bkg_test = np.array(df_bkg_test)
    print 'X_bkg_test.shape: ', X_bkg_test.shape

    y_bkg_test = np.array(X_bkg_test.shape[0] * [0])
    print 'y_bkg_test.shape: ', y_bkg_test.shape


    X_test = np.concatenate((X_sig_test, X_bkg_test))
    y_test = np.concatenate((y_sig_test, y_bkg_test))

    predicted_test = model.predict(X_test)
    #print predicted_test
    predicted_test_clever = (predicted_test + y_test).tolist()
    error_test = float(predicted_test_clever.count(1)) / float(len(predicted_test_clever))
    print "Error: ", error_test

    print "Accuracy_score: ", metrics.accuracy_score(y_test, predicted_test)
    print "ROC_auc_score: ", metrics.roc_auc_score(y_test, predicted_test)


    # 6. Using input a new value    
    result = model.predict_proba(np.array([10,1,2]))
    print 'model.predict_proba(np.array([16,1,3])): ', result #(np.array([20,5, 100, 400, 25]))
    #print 'model.predict_proba(np.array([20,5])): ', model.predict_proba(scaler.transform([[20,5]]))

    ########################################################### 
    
    return (model, X, y, result, model.score(X, y), error_test)
    #return (model, X_scaled, y)

    
    
    
    


# In[179]:

def visualModel(m, X, y):

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].

    #x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    #y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    #z_min, z_max = X[:, 2].min() - .5, X[:, 2].max() + .5

    x_min, x_max = X[:, 0].min() , X[:, 0].max() 
    y_min, y_max = X[:, 1].min() , X[:, 1].max() 
    z_min, z_max = X[:, 2].min() , X[:, 2].max() 


    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1), np.arange(z_min, z_max, 1))
    print xx

    Z = m.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    #Z = m.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel()*yy.ravel(), xx.ravel()*xx.ravel(), yy.ravel()*yy.ravel()])

    print Z

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    fig = plt.figure(1, figsize=(4, 3))

    ax = fig.add_subplot(111)
    #ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    #ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, edgecolors='k', cmap=plt.cm.Paired)

    #plt.xlabel('PTS')
    #plt.ylabel('AST')
    #plt.ylabel('REB')

    ax.set_xlabel('Assists')
    ax.set_ylabel('Points')
    #ax.set_zlabel('Rebounds')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_zlim(zz.min(), zz.max())

    #plt.xlim(xx.min(), xx.max())
    #plt.ylim(yy.min(), yy.max())
    #plt.zlim(zz.min(), zz.max())
    #plt.xticks(())
    #plt.yticks(())
    #plt.zticks(())

    plt.show()


# In[180]:

def visualProfile(m, X, y):

    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
    Z = m.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure(1, figsize=(4, 3))
    pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)
    
    # Plot also the training points
    pl.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=pl.cm.Paired)
    pl.xlabel('Assists')
    pl.ylabel('Points')
    
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.xticks(())
    pl.yticks(())
    
    pl.show()


def probModel(games_played, prob):
    
    # 1. Fixing player performance, probability of being an all-star as a function of time
    fig = plt.figure()    
    ax = fig.add_subplot(111)
    ax.text(20, 0.2, 'Fixing Player Performance At: PTS=8, AST=1, REB=3', fontsize=10)
    
    d_p = {'ngames' : games_played,
          'prob' : prob}   
    df_p = pd.DataFrame(d_p)
    #df_p.plot(x='ngames', y='prob')  
    
    x = []
    y = []
    x = np.array(df_p['ngames'])
    y = np.array(df_p['prob'])
    
    z = np.polyfit(x, y, 4)
    f = np.poly1d(z)
    
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)
    
    
    plt.plot(x, y,'b^', x_new, y_new, 'r', linewidth=2.0)
    plt.plot(x, y,'b^')
    plt.xlim([x[0]-1, x[-1] + 1 ])
    plt.ylim([0, 1 ])
    plt.xlabel('Number of Games Played')
    plt.ylabel('Probability of Being in the All-star Team')
    plt.grid(True)
    

    mpl.style.use('ggplot')
    plt.savefig("Ngames_Prob", format='pdf')


# In[181]:

def overtrainModel(games_played, misc_ratio):
    
    # 2. Misclassification rate as a function of time
    fig = plt.figure()    
    ax = fig.add_subplot(111)
    ax.text(400, 0.98, 'Overtrain', fontsize=10)
    #ax.text(0, -0.01, 'Overtrain', fontsize=10)
    
    d_p = {'ngames' : games_played,
          'misc_ratio' : misc_ratio}   
    df_p = pd.DataFrame(d_p)
    #df_p.plot(x='ngames', y='prob')  
    
    x = []
    y = []
    x = np.array(df_p['ngames'])
    y = np.array(df_p['misc_ratio'])
    
    z = np.polyfit(x, y, 4)
    f = np.poly1d(z)
    
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)
    
    
    plt.plot(x, y,'gs', x_new, y_new, 'r', linewidth=2.0)
    plt.plot(x, y,'gs')
    plt.xlim([x[0]-1, x[-1] + 1 ])
    plt.ylim([0.9, 1.1])
    #plt.ylim([-0.1, 0.1])
    plt.xlabel('Number of Games Played')
    plt.ylabel('Training / Testing Misclassification Rate')
    
    ax.text(20, 0.98, 'Overtrain', fontsize=10)
    
    plt.grid(True)
    
    plot([x[0]-1, x[-1]+1], [1, 1], color='black', linestyle='-', linewidth=1)
    
    mpl.style.use('ggplot')
    plt.savefig("Ngames_Overtrain", format='pdf')


# In[182]:

def powerModel(games_played, train_score):
    
    # 2. Misclassification rate as a function of time
    fig = plt.figure()    
    ax = fig.add_subplot(111)
    ax.text(20, 0.45, 'Null Prediction Accuracy', fontsize=10)
    
    d_p = {'ngames' : games_played,
          'train_score' : train_score}   
    df_p = pd.DataFrame(d_p)
    #df_p.plot(x='ngames', y='prob')  
    
    x = []
    y = []
    x = np.array(df_p['ngames'])
    y = np.array(df_p['train_score'])
    
    z = np.polyfit(x, y, 4)
    f = np.poly1d(z)
    
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)
    
    
    plt.plot(x, y,'ko', x_new, y_new, 'r', linewidth=2.0)
    plt.plot(x, y,'ko')
    plt.xlim([x[0]-1, x[-1] + 1 ])
    plt.ylim([0, 1])
    plt.xlabel('Number of Games Played')
    plt.ylabel('Prediction Accuracy')
    plt.grid(True)
    
    plot([x[0]-1, x[-1]+1], [0.5, 0.5], color='black', linestyle='-', linewidth=1)
    
    mpl.style.use('ggplot')
    plt.savefig("Ngames_Power", format='pdf')


# In[183]:

def misclassModel(games_played, misc_rate):
    
    # 2. Misclassification rate as a function of time
    fig = plt.figure()    
    ax = fig.add_subplot(111)

    #ax.text(20, 0.2, 'Fixing Player Performance At: PTS=20, AST=6, REB=5', fontsize=10)
    
    d_p = {'ngames' : games_played,
          'misc_rate' : misc_rate}   
    df_p = pd.DataFrame(d_p)
    #df_p.plot(x='ngames', y='prob')  
    
    x = []
    y = []
    x = np.array(df_p['ngames'])
    y = np.array(df_p['misc_rate'])
    
    z = np.polyfit(x, y, 4)
    f = np.poly1d(z)
    
    x_new = np.linspace(x[0], x[-1], 50)
    y_new = f(x_new)
    
    
    plt.plot(x, y,'ms', x_new, y_new, 'r', linewidth=2.0)
    plt.xlim([x[0]-1, x[-1] + 1 ])
    plt.ylim([0.25, 0.35])
    plt.xlabel('Number of Games Played')
    plt.ylabel('Misclassification Rate')
    plt.grid(True)
    

    mpl.style.use('ggplot')
    plt.savefig("Ngames_Misc", format='pdf')


# In[184]:

def main():

    # preprocessing data
    df = {}

    #df['sig'] = pd.read_csv('allstar_post1980_sql_log.csv',sep='\t')
    #df['bkg'] = pd.read_csv('nonstar_post1980_sql_log.csv',sep='\t')


    ## features: a, b, c ...
    ## conditions: 1. has to be <= X games the user input    2. has to be <= Y games played since the first all-star

    #df['sig_10g'] = df['sig'][["PTS","AST","REB"]][(df['sig']["PLAYED"] <=10) & (df['sig']["LEARN_INDICATOR"] == 1) ]
    #df['bkg_10g'] = df['bkg'][["PTS","AST","REB"]][(df['bkg']["PLAYED"] <=10) & (df['bkg']["LEARN_INDICATOR"] == 1) ]   # need randomly sampling probably???

    #df['sig_50g'] = df['sig'][["PTS","AST","REB"]][(df['sig']["PLAYED"] <=50) & (df['sig']["LEARN_INDICATOR"] == 1) ]
    #df['bkg_50g'] = df['bkg'][["PTS","AST","REB"]][(df['bkg']["PLAYED"] <=50) & (df['bkg']["LEARN_INDICATOR"] == 1) ]   # need randomly sampling probably???

    #df['sig_100g'] = df['sig'][["PTS","AST","REB"]][(df['sig']["PLAYED"] <=100) & (df['sig']["LEARN_INDICATOR"] == 1) ]
    #df['bkg_100g'] = df['bkg'][["PTS","AST","REB"]][(df['bkg']["PLAYED"] <=100) & (df['bkg']["LEARN_INDICATOR"] == 1) ]   # need randomly sampling probably???


    # now working on the database version
    #games_played = [1, 2, 3,4,5] 
    #games_played = [5, 10, 20, 50, 80, 120, 160, 200, 250, 300, 350, 400, 500]
    #games_played = [10, 20, 50, 100, 160, 200, 250, 300, 350, 400, 500]
    games_played = [100]


    db = mdb.connect(user="root", host="localhost", db="gamelogdb", charset='utf8')

    cur = {}
    
    prob = []
    train_s = []
    train_o = []
    mis_r = []
    
    for n in games_played:
        
        sig_str = 'sig_%dg' % n
        bkg_str = 'bkg_%dg' % n
        
        cur[sig_str]  = db.cursor(mdb.cursors.DictCursor)
        cur[sig_str].execute("SELECT PTS,AST,REB FROM star WHERE PLAYED<=%d AND LEARN_INDICATOR=1;" % n)
        df[sig_str] = pd.DataFrame( cur[sig_str].fetchall() )

        cur[bkg_str]  = db.cursor(mdb.cursors.DictCursor)
        cur[bkg_str].execute("SELECT PTS,AST,REB FROM non_star WHERE PLAYED<=%d AND LEARN_INDICATOR=1;" % n)
        df[bkg_str] = pd.DataFrame( cur[bkg_str].fetchall() )




        train_sig = {}
        train_bkg = {}
        # Signal: use 80% for training, 20% for testing
        # Background: use sig*5 for training, rest for testing 
        train_sig[sig_str] = random.sample(df[sig_str].index, int(df[sig_str].shape[0] * 0.6)) #int(df['sig_10g'].shape[0] * 0.8))
        train_bkg[bkg_str] = random.sample(df[bkg_str].index, len(train_sig[sig_str]))#int(df[bkg_str].shape[0] * 0.6)) #int(df['sig_10g'].shape[0] * 0.8 * 5))

        df[sig_str+'_training'] = df[sig_str].ix[ train_sig[sig_str] ]
        df[bkg_str+'_training'] = df[bkg_str].ix[ train_bkg[bkg_str] ]

        df[sig_str+'_testing'] = df[sig_str].drop(train_sig[sig_str])
        #df[bkg_str+'_testing'] = df[bkg_str].drop(train_bkg[bkg_str])
        df[bkg_str+'_intermediate'] = df[bkg_str].drop(train_bkg[bkg_str])
        df[bkg_str+'_testing'] = df[bkg_str].ix[ random.sample(df[bkg_str+'_intermediate'].index, int(df[sig_str+'_testing'].shape[0])) ]
        

        print "df['sig_%dg'].shape: " % n,  df[sig_str].shape
        print "df['bkg_%dg'].shape: " % n,  df[bkg_str].shape
        print "df['sig_%dg_training'].shape: " % n,  df[sig_str+'_training'].shape
        print "df['bkg_%dg_training'].shape: " % n,  df[bkg_str+'_training'].shape
        print "df['sig_%dg_testing'].shape: " % n,   df[sig_str+'_testing'].shape
        print "df['bkg_%dg_testing'].shape: " % n,   df[bkg_str+'_testing'].shape

         
        (trained_model, data, label, result, train_score, misc_rate) = regressionModel(df[sig_str+'_training'], df[bkg_str+'_training'], df[sig_str+'_testing'], df[bkg_str+'_testing'])
    #regressionModel(df['sig_10g_training'], df['bkg_10g_training'], df['sig_10g_testing'], df['bkg_10g_testing'])
    #(trained_model, data, label) = regressionModel(df['sig_50g_training'], df['bkg_50g_training'], df['sig_50g_testing'], df['bkg_50g_testing'])
    #regressionModel(df['sig_100g_training'], df['bkg_100g_training'], df['sig_100g_testing'], df['bkg_100g_testing'])
        
        prob.append(result[0][1])
        train_s.append(train_score)
        train_o.append(float(train_score)/float(1-misc_rate))
        mis_r.append(misc_rate)
        
    #visualModel(trained_model, data, label)    
    #visualProfile(trained_model, data, label)    
    
    #probModel(games_played, prob)

    #overtrainModel(games_played, train_o)
    #
    #powerModel(games_played, train_s)
    #
    #misclassModel(games_played, mis_r)
    
    
    # visualizing results here:
    
    #fig1 = plt.figure()    
    #ax = fig1.add_subplot(111)    
    #ax.set_xlabel('Number of Points (PTS)')
    #ax.set_ylabel('Arbitrary Unit')
    #ax.text(9, 0.05, 'Selected Player', fontsize=10)
    #df['sig_500g_training']['PTS'].hist(normed=True, bins=20, range=(0,40), alpha=0.4)
    #df['bkg_500g_training']['PTS'].hist(normed=True, bins=20, range=(0,40), alpha=0.4)
    #plot([8, 8], [0, 0.09], color='black', linestyle='-', linewidth=1)
    #plt.savefig("pts_dist", format='pdf')
    #
    #fig2 = plt.figure()
    #ax = fig2.add_subplot(111)
    #ax.set_xlabel('Number of Assists (AST)')
    #ax.set_ylabel('Arbitrary Unit') 
    #ax.text(1.5, 0.5, 'Selected Player', fontsize=10)
    #df['sig_500g_training']['AST'].hist(normed=True, bins=20, range=(0,20), alpha=0.4)
    #df['bkg_500g_training']['AST'].hist(normed=True, bins=20, range=(0,20), alpha=0.4)
    #plot([1, 1], [0, 0.7], color='black', linestyle='-', linewidth=1)
    #plt.savefig("ast_dist", format='pdf')
    #
    #fig3 = plt.figure()
    #ax = fig3.add_subplot(111)
    #ax.set_xlabel('Number of Rebounds (REB)')
    #ax.set_ylabel('Arbitrary Unit')  
    #ax.text(3.5, 0.4, 'Selected Player', fontsize=10)
    #df['sig_500g_training']['REB'].hist(normed=True, bins=20, range=(0,20), alpha=0.4)
    #df['bkg_500g_training']['REB'].hist(normed=True, bins=20, range=(0,20), alpha=0.4)
    #plot([3, 3], [0, 0.6], color='black', linestyle='-', linewidth=1)
    #plt.savefig("bkg_dist", format='pdf')    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()


    


# In[120]:




# In[ ]:



