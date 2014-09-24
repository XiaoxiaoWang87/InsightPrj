#!/usr/bin/python
import sys
import os
import csv
import time
import datetime
from types import *

import random
import math

import pandas as pd
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl

import pymysql as mdb
from scipy import special

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


    grouped = {}
    grouped['2013'] = df['2013'].groupby('Player_ID')


    count = 0
    for name, g in grouped['2013']:

        #if count>0:
        #    continue
        count = count+1

        n = g.shape[0]
        m = g.mean()
        print name, n, m

        #if int(m["Player_ID"])!= 203506:
        #    continue

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

       
        # Distributions
        #features = ['PTS','AST','REB','STL','TOV','BLK','FGA','FGM','FTA','FTM','WL']

        features = ['PTS','AST','FG_PCT','REB','MIN','STL','BLK','EFF']
        for f in features:
            mpl.style.use('ggplot')
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            if f =='PTS':
                xhigh = 40
                xlab = 'Points (PTS)'
            elif f =='MIN': 
                xhigh = 40
                xlab = 'Minutes Played (MIN)'
            elif f=='EFF':
                xhigh = 40
                xlab = 'Efficiency Rating (EFF)'
            elif f=='AST':
                xhigh = 10
                xlab = 'Assists (AST)'
            elif f=='REB':
                xhigh = 10
                xlab = 'Rebounds (REB)'
            elif f=='STL':
                xhigh = 6
                xlab = 'Steals (STL)'
            elif f=='BLK':
                xhigh = 6
                xlab = 'Blocked Shots (BLK)'
            else:
                xhigh = 1
                xlab = 'Field Goal Percentage (FG_PCT)'
            #plt.xlim([0.0, xhigh])
            #plt.ylim([0.0, 1.0])
            #hist_sig = df[sig_str][f].hist(normed=True, bins=10, range=(0,xhigh), alpha=0.4, color='r')
            #hist_bkg = df[bkg_str][f].hist(normed=True, bins=10, range=(0,xhigh), alpha=0.4, color='b')   
            hist_sig = np.histogram(df[sig_str][f].as_matrix(),bins=10,range=(0,xhigh), density=True)
            hist_bkg = np.histogram(df[bkg_str][f].as_matrix(),bins=10,range=(0,xhigh), density=True)

            bin_edges = hist_sig[1]
            bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
            bin_widths = (bin_edges[1:] - bin_edges[:-1])


            ax.set_xlabel(xlab, fontsize=16)
            ax.set_ylabel('Arbitrary Unit', fontsize=16)
            ax.text(m[f]+m[f]*0.05, hist_sig[0].max()*0.4, 'Selected Player', fontsize=15)
            ax.bar(bin_centers-bin_widths/2.,hist_sig[0],facecolor='blue',linewidth=0,width=bin_widths,alpha=0.4, label='All-star Player')
            ax.bar(bin_centers-bin_widths/2.,hist_bkg[0],facecolor='red',linewidth=0,width=bin_widths,alpha=0.4, label='Ordinary Player')

            legend = ax.legend(loc='upper right', shadow=True, ncol=2)

            plt.plot([m[f], m[f]], [0, hist_sig[0].max()*1.2], color='black', linestyle='-', linewidth=1)
            plt.savefig(f+"_"+str(int(m["Player_ID"]))+".jpg")
            plt.close()

if __name__ == '__main__':
    main()


