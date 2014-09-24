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
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn_pandas import DataFrameMapper, cross_val_score

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from mpl_toolkits.mplot3d import Axes3D

from patsy import dmatrices

import pymysql as mdb


def bdtModel(df_sig_train, df_bkg_train, df_sig_test, df_bkg_test, sr):

    # '---------- Prepare Training ----------'

    X_sig = np.array(df_sig_train)
    y_sig = np.array(X_sig.shape[0] * [1])
    X_bkg = np.array(df_bkg_train)
    y_bkg = np.array(X_bkg.shape[0] * [0])

    X = np.concatenate((X_sig, X_bkg))
    y = np.concatenate((y_sig, y_bkg))

    print 'X_sig.shape: ', X_sig.shape
    print 'y_sig.shape: ', y_sig.shape
    print 'X_bkg.shape: ', X_bkg.shape
    print 'y_bkg.shape: ', y_bkg.shape
    print 'X.shape: ', X.shape
    print 'y.shape: ', y.shape

    # '---------- Prepare Testing ----------'

    X_sig_test = np.array(df_sig_test)
    y_sig_test = np.array(X_sig_test.shape[0] * [1])
    X_bkg_test = np.array(df_bkg_test)
    y_bkg_test = np.array(X_bkg_test.shape[0] * [0])

    X_test = np.concatenate((X_sig_test, X_bkg_test))
    y_test = np.concatenate((y_sig_test, y_bkg_test))

    print 'X_sig_test.shape: ', X_sig_test.shape
    print 'y_sig_test.shape: ', y_sig_test.shape
    print 'X_bkg_test.shape: ', X_bkg_test.shape
    print 'y_bkg_test.shape: ', y_bkg_test.shape
    print 'X_test.shape: ', X_test.shape
    print 'y_test.shape: ', y_test.shape


    # '---------- Model ----------'

    #scaler = preprocessing.StandardScaler().fit(X)
    #X = scaler.transform(X)

    #model = svm.SVC(C = 50, kernel = 'rbf', tol=0.001, gamma=0.005, probability=True)
    #model.fit(X, y)

    dt = DecisionTreeClassifier(max_depth=3,
                                min_samples_leaf=0.05*len(X))
    model = AdaBoostClassifier(dt,
                             algorithm='SAMME',
                             n_estimators=800,
                             learning_rate=0.5)
    
    model.fit(X, y)


    print '---------- Training/Testing info ----------'

    print 'Accuracy (training): ', model.score(X, y)
    print 'Null Error Rate (training): ', y.mean()


    #X_test = scaler.transform(X_test)
    predicted_test = model.predict(X_test)

    predicted_test_clever = (predicted_test + y_test).tolist()
    error_test = float(predicted_test_clever.count(1)) / float(len(predicted_test_clever))
    print "Error: ", error_test

    print "Accuracy (testing): ", metrics.accuracy_score(y_test, predicted_test)
    print "Recall (testing): ",   metrics.recall_score(y_test, predicted_test)
    print "F1 score (testing): ", metrics.f1_score(y_test, predicted_test)
    print "ROC area under curve (testing): ", metrics.roc_auc_score(y_test, predicted_test)

    #user_input = scaler.transform(np.array([10, 1, 2, 0, 2, 0.3, 0.3, 0.3, 10, 5, 1], dtype=float))
    #user_input = scaler.transform(np.array([10,1,2,2,2,2,2,2,2,2,1], dtype=float))
    #user_input = scaler.transform(np.array([10,1,2], dtype=float))

    user_input = np.array([sr['PTS'],sr['AST'],sr['REB'],sr['STL'],sr['BLK'],sr['FG_PCT'],sr['FG3_PCT'],sr['FT_PCT'],sr['MIN'],sr['EFF'],sr['WL']], dtype=float)
    #user_input = np.array([10,1,2,2,2,2,2,2,2,2,1], dtype=float)
    print user_input
    score = model.decision_function(user_input)
    print 'Score (user input): ', score
    result = model.predict_proba(user_input)
    print 'Probability of 1 (user input): ', result



    # '--------- Visualization -----------'

    #Classifier_training_S = model.decision_function(X[y>0.5]).ravel()
    #Classifier_training_B = model.decision_function(X[y<0.5]).ravel()
    #Classifier_testing_S = model.decision_function(X_test[y_test>0.5]).ravel()
    #Classifier_testing_B = model.decision_function(X_test[y_test<0.5]).ravel()

    #(h_test_s, h_test_b) =  visualSigBkg("BDT", Classifier_training_S, Classifier_training_B, Classifier_testing_S, Classifier_testing_B)
    ########################################################### 

    #return (model, X, y, result, model.score(X, y), error_test, h_test_s, h_test_b)
    return (model, X, y, result, model.score(X, y), error_test, score)



def visualSigBkg(model_name, Classifier_training_S, Classifier_training_B, Classifier_testing_S, Classifier_testing_B):

    c_max = 2.0
    c_min = -2.0

    if model_name=="BDT":
        c_max = 0.4
        c_min = -0.3 
        #c_max = 1.0
        #c_min = -0.8

    Histo_training_S_ori = np.histogram(Classifier_training_S,bins=20,range=(c_min,c_max))
    Histo_training_B_ori = np.histogram(Classifier_training_B,bins=20,range=(c_min,c_max))
    Histo_testing_S_ori = np.histogram(Classifier_testing_S,bins=20,range=(c_min,c_max))
    Histo_testing_B_ori = np.histogram(Classifier_testing_B,bins=20,range=(c_min,c_max))

    Histo_training_S = np.histogram(Classifier_training_S,bins=20,range=(c_min,c_max), density=True)
    Histo_training_B = np.histogram(Classifier_training_B,bins=20,range=(c_min,c_max), density=True)
    Histo_testing_S = np.histogram(Classifier_testing_S,bins=20,range=(c_min,c_max), density=True)
    Histo_testing_B = np.histogram(Classifier_testing_B,bins=20,range=(c_min,c_max), density=True)

    Histo_training_S_SF = Histo_training_S[0].sum() / Histo_training_S_ori[0].sum() 
    Histo_training_B_SF = Histo_training_B[0].sum() / Histo_training_B_ori[0].sum()
    Histo_testing_S_SF  = Histo_testing_S[0].sum() / Histo_testing_S_ori[0].sum()
    Histo_testing_B_SF  = Histo_testing_B[0].sum() / Histo_testing_B_ori[0].sum()


    # Lets get the min/max of the Histograms
    AllHistos= [Histo_training_S, Histo_training_B, Histo_testing_S, Histo_testing_B]
    h_max = max([histo[0].max() for histo in AllHistos])*1.5
    h_min = max([histo[0].min() for histo in AllHistos])

    # Get the histogram properties (binning, widths, centers)
    bin_edges = Histo_training_S[1]
    bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    # To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
    ErrorBar_testing_S = np.sqrt(Histo_testing_S_ori[0]) * Histo_testing_S_SF 
    ErrorBar_testing_B = np.sqrt(Histo_testing_B_ori[0]) * Histo_testing_B_SF

    plt.figure()

    # Draw objects
    ax1 = plt.subplot(111)

    # Draw solid histograms for the training data
    ax1.bar(bin_centers-bin_widths/2.,Histo_training_S[0],facecolor='blue',linewidth=0,width=bin_widths,label='S (Train)',alpha=0.5)
    ax1.bar(bin_centers-bin_widths/2.,Histo_training_B[0],facecolor='red',linewidth=0,width=bin_widths,label='B (Train)',alpha=0.5)

    # # Draw error-bar histograms for the testing data
    ax1.errorbar(bin_centers, Histo_testing_S[0], yerr=ErrorBar_testing_S, xerr=None, ecolor='blue',c='blue',fmt='o',label='S (Test)')
    ax1.errorbar(bin_centers, Histo_testing_B[0], yerr=ErrorBar_testing_B, xerr=None, ecolor='red',c='red',fmt='o',label='B (Test)')

    # Make a colorful backdrop to show the clasification regions in red and blue
    ax1.axvspan(0.0, c_max, color='blue',alpha=0.08)
    ax1.axvspan(c_min,0.0, color='red',alpha=0.08)

    # Adjust the axis boundaries (just cosmetic)
    ax1.axis([c_min, c_max, h_min, h_max])

    # Make labels and title
    plt.title("Discriminating Power and Overtraining Check")
    plt.xlabel("Classifier, %s" % model_name) #SVM [rbf kernel, C=1, gamma=0.005]")
    plt.ylabel("Arbituary Unit")

    # Make legend with smalll font
    legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
    for alabel in legend.get_texts():
                alabel.set_fontsize('small')

    # Save the result to png
    plt.savefig("Sklearn_disc_"+model_name+".pdf")


    return (Histo_testing_S_ori[0], Histo_testing_B_ori[0])
    #visualROC(model_name, Histo_testing_S_ori[0], Histo_testing_B_ori[0])



def visualROC(model_name, h_s, h_b):

    Color = ['m','r','k','b']
    plt.figure()
    #plt.figure(figsize=(6, 6))
    ax = plt.subplot(111) 
    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('True Positive Rate')
    plt.ylabel('True Negative Rate')
    plt.title('ROC Curve')
    ax.text(0.3, 0.62, "Chance Performance", rotation=-38, fontsize=15)

    for m in range(len(model_name)):
        
        Size = h_s[m].size
        sig_eff = np.zeros(Size)
        bkg_eff = np.zeros(Size)
        bkg_rej = np.zeros(Size)   
    
        for i in range(Size):
    
            tmp_h_s = h_s[m][i:Size]
            tmp_h_b = h_b[m][i:Size]
            sig_eff[i] = float(tmp_h_s.sum())/float(h_s[m].sum())
            bkg_eff[i] = float(tmp_h_b.sum())/float(h_b[m].sum())
            bkg_rej[i] = 1 - bkg_eff[i]
    
        sig_eff.sort()
        bkg_rej = bkg_rej[::-1]
        
        print sig_eff
        print bkg_rej
    
        plt.plot(sig_eff, bkg_rej, label='%s' % model_name[m], color=Color[m], linewidth=2.0)

    plt.legend(loc="lower left")
    plt.savefig("Sklearn_ROC"+".pdf")   


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
    plt.savefig("Ngames_Prob.pdf")


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
    plt.savefig("Ngames_Overtrain.pdf")


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
    plt.savefig("Ngames_Power.pdf")


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
    plt.savefig("Ngames_Misc.pdf")


# In[184]:

def main():

    # preprocessing data
    df = {}


    # now working on the database version
    #games_played = [1, 2, 3,4,5] 
    #games_played = [5, 10, 20, 50, 80, 120, 160, 200, 250, 300, 350, 400, 500]
    #games_played = [10, 20, 50, 100, 160, 200, 250, 300, 350, 400, 500]
    games_played = [100]


    #db = mdb.connect(user="root", host="localhost", db="gamelogdb", charset='utf8')
    db = mdb.connect(user="root", host="localhost", db="nbagamedb", charset='utf8')

    cur = {}
    
    prob = []
    train_s = []
    train_o = []
    mis_r = []
    final = []


    sig_2010 = "sig_2010"
    bkg_2010 = "bkg_2010"
    cur[sig_2010]  = db.cursor(mdb.cursors.DictCursor)
    cur[sig_2010].execute("SELECT PTS,AST,REB,STL,BLK,FGA,FGM,FTA,FTM,TOV,WL,FG_PCT,FG3_PCT,FT_PCT,MIN,Player_ID FROM star WHERE PLAYED<=60 AND FROM_YEAR=2010 AND LEARN_INDICATOR=1;")
    df[sig_2010] = pd.DataFrame( cur[sig_2010].fetchall() )

    cur[bkg_2010]  = db.cursor(mdb.cursors.DictCursor)
    cur[bkg_2010].execute("SELECT PTS,AST,REB,STL,BLK,FGA,FGM,FTA,FTM,TOV,WL,FG_PCT,FG3_PCT,FT_PCT,MIN,Player_ID FROM non_star WHERE PLAYED<=60 AND FROM_YEAR=2010 AND LEARN_INDICATOR=1;")
    df[bkg_2010] = pd.DataFrame( cur[bkg_2010].fetchall() )

    df["2010"] = pd.concat([df[sig_2010], df[bkg_2010]])

    df["2010"]['EFF'] = (df["2010"]['PTS'] + df["2010"]['REB'] + df["2010"]['AST'] + df["2010"]['STL'] + df["2010"]['BLK']) - ((df["2010"]['FGA']-df["2010"]['FGM'])+(df["2010"]['FTA']-df["2010"]['FTM'])+df["2010"]['TOV'])
    df["2010"]['WL'] = df["2010"]['WL'].map(lambda x: 1 if x=='W' else 0)



    sig_2013 = "sig_2013"
    bkg_2013 = "bkg_2013"
    #cur[sig_2013]  = db.cursor(mdb.cursors.DictCursor)
    #cur[sig_2013].execute("SELECT PTS,AST,REB,STL,BLK,FGA,FGM,FTA,FTM,TOV,WL,FG_PCT,FG3_PCT,FT_PCT,MIN FROM star WHERE PLAYED<=60 AND FROM_YEAR=2013;")
    #df[sig_2013] = pd.DataFrame( cur[sig_2013].fetchall() )

    cur[bkg_2013]  = db.cursor(mdb.cursors.DictCursor)
    cur[bkg_2013].execute("SELECT PTS,AST,REB,STL,BLK,FGA,FGM,FTA,FTM,TOV,WL,FG_PCT,FG3_PCT,FT_PCT,MIN,Player_ID FROM non_star WHERE PLAYED<=60 AND FROM_YEAR=2013;")
    df[bkg_2013] = pd.DataFrame( cur[bkg_2013].fetchall() )

    df["2013"] = df[bkg_2013] #pd.concat([df[sig_2013], df[bkg_2013]])

    df["2013"]['EFF'] = (df["2013"]['PTS'] + df["2013"]['REB'] + df["2013"]['AST'] + df["2013"]['STL'] + df["2013"]['BLK']) - ((df["2013"]['FGA']-df["2013"]['FGM'])+(df["2013"]['FTA']-df["2013"]['FTM'])+df["2013"]['TOV'])
    df["2013"]['WL'] = df["2013"]['WL'].map(lambda x: 1 if x=='W' else 0)


    #df["2010"] = df["2010"].groupby(['Player_ID'], as_index=False).mean()
    #df["2013"] = df["2013"].groupby(['Player_ID'], as_index=False).mean()

    #print df["2010"].shape
    #print df["2013"].shape

    grouped = {}
    grouped['2010'] = df['2010'].groupby('Player_ID')       
    grouped['2013'] = df['2013'].groupby('Player_ID')


    count = 0
    for name, g in grouped['2013']:

        #if count>0:
        #    continue
        count = count+1

        n = g.shape[0]
        m = g.mean()
        print name, n, m
        
        sig_str = 'sig_%dg' % n
        bkg_str = 'bkg_%dg' % n
        
        # Efficiency:  (Points + Rebounds + Assists + Steals + Blocks) - ( (Field Goals Att. - Field Goals Made) + (Free Throws Att. -Free Throws Made) + Turnovers ) 
        # Eff = (PTS + REB + AST + STL + BLK) - ( (FGA - FGM) + (FTA - FTM) + TOV )
        # Use players before 2010 to train and test

        cur[sig_str]  = db.cursor(mdb.cursors.DictCursor)
        cur[sig_str].execute("SELECT PTS,AST,REB,STL,BLK,FGA,FGM,FTA,FTM,TOV,WL,FG_PCT,FG3_PCT,FT_PCT,MIN,FROM_YEAR FROM star WHERE PLAYED<=%d AND LEARN_INDICATOR=1;" % n)
        df[sig_str] = pd.DataFrame( cur[sig_str].fetchall() )

        cur[bkg_str]  = db.cursor(mdb.cursors.DictCursor)
        cur[bkg_str].execute("SELECT PTS,AST,REB,STL,BLK,FGA,FGM,FTA,FTM,TOV,WL,FG_PCT,FG3_PCT,FT_PCT,MIN,FROM_YEAR FROM non_star WHERE PLAYED<=%d AND LEARN_INDICATOR=1;" % n)
        df[bkg_str] = pd.DataFrame( cur[bkg_str].fetchall() )

        # calculating effciency
        df[sig_str]['EFF'] = (df[sig_str]['PTS'] + df[sig_str]['REB'] + df[sig_str]['AST'] + df[sig_str]['STL'] + df[sig_str]['BLK']) - ((df[sig_str]['FGA']-df[sig_str]['FGM'])+(df[sig_str]['FTA']-df[sig_str]['FTM'])+df[sig_str]['TOV'])
        df[bkg_str]['EFF'] = (df[bkg_str]['PTS'] + df[bkg_str]['REB'] + df[bkg_str]['AST'] + df[bkg_str]['STL'] + df[bkg_str]['BLK']) - ((df[bkg_str]['FGA']-df[bkg_str]['FGM'])+(df[bkg_str]['FTA']-df[bkg_str]['FTM'])+df[bkg_str]['TOV'])


        # splitting into three sets
        #df[sig_str+'_2010'] = df[sig_str][ df[sig_str]['FROM_YEAR'] == 2010 ]
        #df[bkg_str+'_2010'] = df[bkg_str][ df[bkg_str]['FROM_YEAR'] == 2010 ]

        #df[sig_str+'_2013'] = df[sig_str][ df[sig_str]['FROM_YEAR'] == 2013 ]
        #df[bkg_str+'_2013'] = df[bkg_str][ df[bkg_str]['FROM_YEAR'] == 2013 ]

        df[sig_str] = df[sig_str][ df[sig_str]['FROM_YEAR'] < 2010 ]
        df[bkg_str] = df[bkg_str][ df[bkg_str]['FROM_YEAR'] < 2010 ]



        # trim down variables
        df[sig_str] = df[sig_str].ix[:,['PTS','AST','REB','STL','BLK','FG_PCT','FG3_PCT','FT_PCT','MIN','EFF','WL']]
        df[bkg_str] = df[bkg_str].ix[:,['PTS','AST','REB','STL','BLK','FG_PCT','FG3_PCT','FT_PCT','MIN','EFF','WL']]



        train_sig = {}
        train_bkg = {}
        # Signal: use 80% for training, 20% for testing
        # Background: use sig*5 for training, rest for testing 
        train_sig[sig_str] = random.sample(df[sig_str].index, int(df[sig_str].shape[0] * 0.7)) #int(df['sig_10g'].shape[0] * 0.8))
        train_bkg[bkg_str] = random.sample(df[bkg_str].index, len(train_sig[sig_str]))         #int(df[bkg_str].shape[0] * 0.6)) #int(df['sig_10g'].shape[0] * 0.8 * 5))

        df[sig_str+'_training'] = df[sig_str].ix[ train_sig[sig_str] ]
        df[bkg_str+'_training'] = df[bkg_str].ix[ train_bkg[bkg_str] ]

        df[sig_str+'_testing'] = df[sig_str].drop(train_sig[sig_str])
        #df[bkg_str+'_testing'] = df[bkg_str].drop(train_bkg[bkg_str])
        df[bkg_str+'_intermediate'] = df[bkg_str].drop(train_bkg[bkg_str])
        df[bkg_str+'_testing'] = df[bkg_str].ix[ random.sample(df[bkg_str+'_intermediate'].index, int(df[sig_str+'_testing'].shape[0])) ]
 
        df[sig_str+'_training']['WL'] = df[sig_str+'_training']['WL'].map(lambda x: 1 if x=='W' else 0)
        df[bkg_str+'_training']['WL'] = df[bkg_str+'_training']['WL'].map(lambda x: 1 if x=='W' else 0)
        df[sig_str+'_testing']['WL'] = df[sig_str+'_testing']['WL'].map(lambda x: 1 if x=='W' else 0)
        df[bkg_str+'_testing']['WL'] = df[bkg_str+'_testing']['WL'].map(lambda x: 1 if x=='W' else 0)


        print "df['sig_%dg'].shape: " % n,  df[sig_str].shape
        print "df['bkg_%dg'].shape: " % n,  df[bkg_str].shape
        print "df['sig_%dg_training'].shape: " % n,  df[sig_str+'_training'].shape
        print "df['bkg_%dg_training'].shape: " % n,  df[bkg_str+'_training'].shape
        print "df['sig_%dg_testing'].shape: " % n,   df[sig_str+'_testing'].shape
        print "df['bkg_%dg_testing'].shape: " % n,   df[bkg_str+'_testing'].shape


        #(trained_model, data, label, result, train_score, misc_rate, h1_test_sig, h1_test_bkg) = regressionModel(df[sig_str+'_training'], df[bkg_str+'_training'], df[sig_str+'_testing'], df[bkg_str+'_testing'])
        #(trained_model, data, label, result, train_score, misc_rate, h2_test_sig, h2_test_bkg) = svmModel(df[sig_str+'_training'], df[bkg_str+'_training'], df[sig_str+'_testing'], df[bkg_str+'_testing'])
        (trained_model, data, label, result, train_score, misc_rate, score) = bdtModel(df[sig_str+'_training'], df[bkg_str+'_training'], df[sig_str+'_testing'], df[bkg_str+'_testing'], m)

        #C = 10.0 ** np.arange(-2, 9) #[0.01, 0.1, 1.0, 10, 100, 1000, 10000]
        #tol= [0.001] #[0.1, 0.01, 0.001, 0.0001] #[0.1, 0.01, 0.001, 0.0001, 0.00001]
        #gamma= 10.0 ** np.arange(-5, 4)  #[0.05, 0.005, 0.5, 5, 0.0005]

        #for c in C:
        #    for t in tol:
        #        for g in gamma:
        #(trained_model, data, label, result, train_score, misc_rate, h2_test_sig, h2_test_bkg) = svmModel(df[sig_str+'_training'], df[bkg_str+'_training'], df[sig_str+'_testing'], df[bkg_str+'_testing'])


        prob.append(result[0][1])
        train_s.append(train_score)
        train_o.append(float(train_score)/float(1-misc_rate))
        mis_r.append(misc_rate)
         

        #final.append((int(m['Player_ID']), result[0][1]))   
        final.append((int(m['Player_ID']), score))

        #visualModel(trained_model, data, label)    
        #visualProfile(trained_model, data, label)    
        
        #probModel(games_played, prob)

        #overtrainModel(games_played, train_o)
        #
        #powerModel(games_played, train_s)
        #
        #misclassModel(games_played, mis_r)
    final.sort(key=lambda tup: tup[1], reverse=True)       
    print final
        # visualizing results here:
 
        #fig = plt.figure()    
        #ax = fig.add_subplot(111)    
        #ax.set_xlabel('Efficiency (EFF)')
        #ax.set_ylabel('Arbitrary Unit')
        #ax.text(9, 0.05, 'Selected Player', fontsize=10)
   
        #sig_str = 'sig_100g'
        #bkg_str = 'bkg_100g'
        ## Eff = (PTS + REB + AST + STL + BLK) - ( (FGA - FGM) + (FTA - FTM) + TOV )
        #df[sig_str+'_testing']['EFF'].hist(normed=True, bins=20, range=(0,40), alpha=0.4)
        #df[bkg_str+'_testing']['EFF'].hist(normed=True, bins=20, range=(0,40), alpha=0.4)

        #h3_test_sig =  np.histogram(np.array(df[sig_str+'_testing']['EFF'].tolist()),bins=20,range=(0,40))
        #h3_test_bkg =  np.histogram(np.array(df[bkg_str+'_testing']['EFF'].tolist()),bins=20,range=(0,40))



        ## ROC curve
        #plt.plot([8, 8], [0, 0.09], color='black', linestyle='-', linewidth=1)
        #plt.savefig("eff_dist.pdf")

        #hlist_test_sig = [h4_test_sig, h4_test_sig, h1_test_sig, h3_test_sig[0]]
        #hlist_test_bkg = [h4_test_bkg, h4_test_bkg, h1_test_bkg, h3_test_bkg[0]]
        #model_name_list = ['BDT','SVM', 'Logistic Regression', 'NBA Player Efficiency']
        #
        #visualROC(model_name_list, hlist_test_sig, hlist_test_bkg)



        # Distributions
        #features = ['PTS','AST','REB','STL','TOV','BLK','FGA','FGM','FTA','FTM','WL']
        #features = ['PTS','AST','REB','STL','BLK','FG_PCT','FG3_PCT','FT_PCT','MIN','EFF','WL']
        #for f in features:
        #    fig = plt.figure() 
        #    ax = fig.add_subplot(111)
        #    ax.set_xlabel(f)
        #    ax.set_ylabel('Arbitrary Unit')
        #    ax.text(9, 0.05, 'Selected Player', fontsize=10)
        #    df['sig_100g_training'][f].hist(normed=True, bins=30, range=(0,30), alpha=0.4)
        #    df['bkg_100g_training'][f].hist(normed=True, bins=30, range=(0,30), alpha=0.4)   
        #    plt.plot([8, 8], [0, 0.09], color='black', linestyle='-', linewidth=1)
        #    plt.savefig(f+"_dist.pdf")

    
    
if __name__ == '__main__':
    main()


    


# In[120]:




# In[ ]:



