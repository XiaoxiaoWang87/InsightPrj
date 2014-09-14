#!/usr/bin/python
import sys
import os
import csv
import time
import datetime
from types import *

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import pylab as pl
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn_pandas import DataFrameMapper, cross_val_score

from mpl_toolkits.mplot3d import Axes3D



def prepareSQL(f, grouped):
    i = 0
    for name, group in grouped:
        print float(name)

        #print(group)
        #if i <1:
        #    #group['GAME_DATE'].map(lambda x: time.strptime(x, "%m %d, %y")[2] <= group['FIRST_ALLSTAR_YEAR'])
        #    #print group['FIRST_ALLSTAR_YEAR']
        #    #new_group = group[ ( group['GAME_DATE'].map(lambda x: float(x[-4:]) <= group['FIRST_ALLSTAR_YEAR']) ) ]
        #    group['GAME_DATE'].map(lambda x: float(x[-4:]) <= group['FIRST_ALLSTAR_YEAR']) 
        #    print group['GAME_DATE']
        #    #new_group.to_csv('doesitwork_random.csv', sep='\t', index=False)
        #if float(name) != 893:
        #    continue
        #    print group
        #    group.to_csv('doesitwork_jordan.csv', sep='\t', index=False) 
        #if i>=1:
        #    continue
    
        #if i>=1:
        #    continue

        learn_indi = []
        game_m = []
        game_d = []
        game_y = []
        game_t_sum = []
    
        game_n = []
    
        group_s = {}
        for index, row in group.iterrows():
            #game_time = time.strptime(row['GAME_DATE'].replace(row['GAME_DATE'][-4:], row['GAME_DATE'][-2:]), "%b %d, %y")
            #game_year = row['GAME_DATE'][-4:]
            game_time = time.strptime(row['GAME_DATE'], "%b %d, %Y")
            game_year = game_time[0]
    
            if game_year <= row['FIRST_ALLSTAR_YEAR']:
                learn_indi.append(1)
            else:
                learn_indi.append(0)
    
            game_y.append(game_time[0])
            game_m.append(game_time[1])
            game_d.append(game_time[2])
    
            game_t_sum.append(str(game_time[0]) + '-' + str(game_time[1]) + '-' + str(game_time[2]))
    
        group_s['LEARN_INDICATOR'] = pd.Series(np.array(learn_indi), name='LEARN_INDICATOR')
        group_s['YEAR'] = pd.Series(np.array(game_y), name='YEAR')
        group_s['MONTH'] = pd.Series(np.array(game_m), name='MONTH')
        group_s['DAY'] = pd.Series(np.array(game_d), name='DAY')
    
        group_s['TIME'] = pd.Series(np.array(game_t_sum), name='TIME')
    
        group.index = range(len(learn_indi))
        game_n = range(1, len(learn_indi)+1)
    
        group_s['PLAYED'] = pd.Series(np.array(game_n), name='PLAYED')
    
        new_group = pd.concat([group, group_s['LEARN_INDICATOR'], group_s['YEAR'], group_s['MONTH'], group_s['DAY'], group_s['TIME']], axis=1)
    
        new_sorted_group = new_group.sort(['TIME'], ascending=[1])
    
        # re-indexing
        new_sorted_group.index = range(len(learn_indi))
        final_group = pd.concat([new_sorted_group, group_s['PLAYED']], axis=1)
    
        out = f.replace('log','sql_log')

        if i ==0:
            final_group.to_csv(out + '.csv', sep='\t', index=False)
        else:
            final_group.to_csv(out + '.csv', mode='a', sep='\t', header=False, index=False)
    
        i = i+1


def main():

    df = {}
    
    df['allstarinfo'] = pd.read_csv('allstar_log.csv',sep='\t')
    
    allstar_list = []
    allstar_post1980_list = []
    allstar_post1980_dict = {}
    yrs_up_to_first = []
    
    for index, row in df['allstarinfo'].iterrows():
        if row['PLAYER_ID'] not in allstar_list:
            allstar_list.append(row['PLAYER_ID'])
        else:
            continue
        
        if row['SEASON'] < 1980:
            continue
        
        allstar_post1980_list.append(row['PLAYER_ID'])
        # this is bad code, I know
        allstar_post1980_dict[row['PLAYER_ID']] = [row['SEASON']] #[float(row['EXP'].replace("R", "0")), float(row['SEASON'])]
    
    print allstar_post1980_list
        #print row['PLAYER']
    
    for star in allstar_post1980_list:
        print star, allstar_post1980_dict[star][0]
    
    df['player_post1980_log'] = (pd.read_csv('player_post1980_inclusive_log.csv',sep='\t')).fillna(0)
    
    df['allstar_post1980_log'] = df['player_post1980_log'][(df['player_post1980_log']["Player_ID"].isin(allstar_post1980_list))]
    df['nonstar_post1980_log'] = df['player_post1980_log'][(~df['player_post1980_log']["Player_ID"].isin(allstar_post1980_list))]
    
    df['allstar_post1980_log']['FIRST_ALLSTAR_YEAR'] = df['allstar_post1980_log']['Player_ID'].map(lambda x: allstar_post1980_dict[x][0])
    df['nonstar_post1980_log']['FIRST_ALLSTAR_YEAR'] = df['nonstar_post1980_log']['Player_ID'].map(lambda x: 9999) #pd.Series(np.array([9999.]*df['nonstar_post1980_log'].shape[0])) 
    #df['allstar_post1980_log'].to_csv('doesitwork_allstar.csv', sep='\t', index=False)
    #df['nonstar_post1980_log'].to_csv('doesitwork_nonstar.csv', sep='\t', index=False)
    
    
    grouped = {}
    grouped['allstar_post1980_log'] = df['allstar_post1980_log'].groupby('Player_ID')
    grouped['nonstar_post1980_log'] = df['nonstar_post1980_log'].groupby('Player_ID')

    prepareSQL('allstar_post1980_log', grouped['allstar_post1980_log'])
    prepareSQL('nonstar_post1980_log', grouped['nonstar_post1980_log'])


if __name__ == '__main__':
    main()


#i = 0
#
#for name, group in grouped['allstar_post1980_log']:
#    print float(name)
#    #print(group)
#
#    #if i <1:
#    #    #group['GAME_DATE'].map(lambda x: time.strptime(x, "%m %d, %y")[2] <= group['FIRST_ALLSTAR_YEAR'])
#    #    #print group['FIRST_ALLSTAR_YEAR']
#    #    #new_group = group[ ( group['GAME_DATE'].map(lambda x: float(x[-4:]) <= group['FIRST_ALLSTAR_YEAR']) ) ]
#    #    group['GAME_DATE'].map(lambda x: float(x[-4:]) <= group['FIRST_ALLSTAR_YEAR']) 
#    #    print group['GAME_DATE']
#    #    #new_group.to_csv('doesitwork_random.csv', sep='\t', index=False)
#
#    #if float(name) != 893:
#    #    continue
#    #    print group
#    #    group.to_csv('doesitwork_jordan.csv', sep='\t', index=False) 
#
#    #if i>=1:
#    #    continue
#
#    learn_indi = []
#    game_m = []
#    game_d = []
#    game_y = []
#    game_t_sum = []
#
#    game_n = []
#    
#    group_s = {}
#    for index, row in group.iterrows():
#        game_time = time.strptime(row['GAME_DATE'].replace(row['GAME_DATE'][-4:], row['GAME_DATE'][-2:]), "%b %d, %y") 
#        #game_year = row['GAME_DATE'][-4:]
#        game_year = game_time[0]
#
#        if game_year <= row['FIRST_ALLSTAR_YEAR']:
#            learn_indi.append(1)
#        else:
#            learn_indi.append(0)
#
#        game_y.append(game_time[0])
#        game_m.append(game_time[1])
#        game_d.append(game_time[2])
# 
#        game_t_sum.append(str(game_time[0]) + '-' + str(game_time[1]) + '-' + str(game_time[2]))
#
#    group_s['LEARN_INDICATOR'] = pd.Series(np.array(learn_indi), name='LEARN_INDICATOR')
#    group_s['YEAR'] = pd.Series(np.array(game_y), name='YEAR')
#    group_s['MONTH'] = pd.Series(np.array(game_m), name='MONTH')
#    group_s['DAY'] = pd.Series(np.array(game_d), name='DAY')
#
#    group_s['TIME'] = pd.Series(np.array(game_t_sum), name='TIME')
#
#    group.index = range(len(learn_indi))
#    game_n = range(1, len(learn_indi)+1)
#
#    group_s['PLAYED'] = pd.Series(np.array(game_n), name='PLAYED')
#
#    new_group = pd.concat([group, group_s['LEARN_INDICATOR'], group_s['YEAR'], group_s['MONTH'], group_s['DAY'], group_s['TIME']], axis=1)
# 
#    new_sorted_group = new_group.sort(['TIME'], ascending=[1])
#
#    # re-indexing
#    new_sorted_group.index = range(len(learn_indi))
#    final_group = pd.concat([new_sorted_group, group_s['PLAYED']], axis=1)
#
#    if i ==0:
#        final_group.to_csv('doesitwork_jordan.csv', sep='\t', index=False) 
#    else:
#        final_group.to_csv('doesitwork_jordan.csv', mode='a', sep='\t', header=False, index=False) 
#
#    i = i+1        







#allstar_df = pd.read_csv('allstar_gamelog.csv',sep='\t')
#
#MJ_s = allstar_df["FGA"].fillna(0)[(allstar_df["Player_ID"] == 893) & (allstar_df["SEASON_ID"] >= 21984) & (allstar_df["SEASON_ID"] <= 21985)]
#KB_s = allstar_df["FGA"].fillna(0)[(allstar_df["Player_ID"] == 977) & (allstar_df["SEASON_ID"] == 22003) ]
#
#    #print MJ_s
#    #for i, row in MJ_s:
#    #    print row['FGA']
#
#    #print MJ_s.mean()
#    #print KB_s.mean()
#
##plt.figure()
#
#MJ_s.hist(normed=True, bins=10, range=(0,50), alpha=0.4)
##KB_s.hist(normed=True, bins=10, range=(0,50), alpha=0.4)
#
#    #plt.annotate('local max', xy=(20, 0.05))
#
#
#sb_df = pd.read_csv('player_gamelog.csv',sep='\t')
#SB_s = sb_df["FGA"][(sb_df["Player_ID"] == 76001) & (sb_df["SEASON_ID"] >= 21990) & (sb_df["SEASON_ID"] <= 21992)]
#SB_s.hist(normed=True, bins=10, range=(0,50), alpha=0.4)
#
#
##MJ_df = pd.DataFrame(MJ_s, columns=['FGA'])
##SB_df = pd.DataFrame(SB_s, columns=['FGA'])
#
#MJ_df = allstar_df[["FGA","AST","REB"]][(allstar_df["Player_ID"] == 893) & (allstar_df["SEASON_ID"] >= 21984) & (allstar_df["SEASON_ID"] <= 21985)]
#SB_df = sb_df[["FGA","AST","REB"]][(sb_df["Player_ID"] == 76001) & (sb_df["SEASON_ID"] >= 21990) & (sb_df["SEASON_ID"] <= 21992)]
#
#
#    #print MJ_df
#    #print SB_df
#
#X_sig = array(MJ_df.fillna(0))
##print X_sig
#    #X_sig = MJ_s.values[:]
#print X_sig.shape
#y_sig = np.array(X_sig.shape[0] * [1])
##print y_sig
#print y_sig.shape
#
#X_bkg = array(SB_df.fillna(0))
#    #print X_bkg
#    #X_bkg = SB_s.values[:]
#print X_bkg.shape
#y_bkg = np.array(X_bkg.shape[0] * [0])
#print y_bkg.shape
#
#
#X = np.concatenate((X_sig, X_bkg))
#y = np.concatenate((y_sig, y_bkg))
#
#print X.shape
#print y.shape
#model = LogisticRegression()
#model.fit(X, y)
#print model.score(X, y)
#print y.mean()
#
#print model.predict_proba(np.array([20,2,1]))
#
#
#
#
#allstarinfo_df = pd.read_csv('allstar_log.csv',sep='\t')
#
#allstar_list = []
#yrs_up_to_first = []
#for index, row in allstarinfo_df.iterrows():
#    if row['PLAYER_ID'] not in allstar_list:
#        allstar_list.append(row['PLAYER_ID'])
#    else:
#        continue
#    
#    if row['SEASON'] < 1980:
#        continue
#        
#    #print row['PLAYER']
#    
#    yrs_up_to_first.append(float(row['EXP'].replace("R", "0")))
#    
#plt.figure()
#
#print len(yrs_up_to_first)
#yrs_up_to_first_s = pd.Series(np.array(yrs_up_to_first))
#yrs_up_to_first_s.hist(normed=False, bins=15, range=(0,15), alpha=0.4)
