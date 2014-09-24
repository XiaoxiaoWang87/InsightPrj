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
from sklearn.metrics.pairwise import euclidean_distances


def main():

    # preprocessing data
    df = {}

    games_played = [60]

    db = mdb.connect(user="root", host="localhost", db="nbagamedb", charset='utf8')

    cur = {}

    with open("../json/names.json", 'r') as file_handle:
        names = json.load(file_handle)
        all_players = pd.DataFrame(names['resultSets'][0]['rowSet'], columns=names['resultSets'][0]['headers'])


    result_tuple = [(203487,  0.18595585), (203506,  0.11655443), (203504,  0.05306283), (203486,  0.02186713), (203508,  0.00706554), (203527,  0.00638566), (203480,  0.00477143), (203500,  0.00192156), (203507, -0.00350226), (203476, -0.01304738), (203489, -0.01324894), (203548, -0.01504922), (203474, -0.01980168), (203268, -0.02474724), (203491, -0.03426916), (203473, -0.03692819), (203459, -0.04018171), (203477, -0.04141347), (203505, -0.0438259), (203519, -0.04384942), (202620, -0.04415433), (201979, -0.04804931), (203496, -0.04972171), (203471, -0.04981582), (203503, -0.05217203), (203482, -0.05485713), (203462, -0.05486091), (203485, -0.05782554), (203810, -0.05834132), (203469, -0.05904961), (203539, -0.05988436), (203136, -0.06093478), (203147, -0.06270075), (203120, -0.06496599), (202779, -0.067018), (203315, -0.06791735), (203492, -0.07292787), (203495, -0.07407107), (203501, -0.07426697), (203517, -0.07717779), (203513, -0.07819452), (202197, -0.07968543), (203543, -0.08127428), (203490, -0.08199551), (203544, -0.08233529), (203461, -0.08248828), (203540, -0.08391612), (203499, -0.08454515), (203479, -0.08738991), (203498, -0.08832967), (203468, -0.08932379), (203460, -0.09485895), (203183, -0.09616857), (203458, -0.096383), (203318, -0.09687138), (203497, -0.09712607), (203816, -0.09742036), (203524, -0.09755825), (203502, -0.09786547), (203569, -0.09961734), (203515, -0.1003242), (203561, -0.10408864), (203463, -0.109834), (203484, -0.11683215), (203263, -0.11898496), (203552, -0.12360296), (203488, -0.12583992), (202091, -0.12839389), (203545, -0.13250706), (203546, -0.13418365), (203493, -0.13682664), (203133, -0.13985063), (203467, -0.14074716), (203481, -0.14116683), (203584, -0.14441901), (203521, -0.17158767), (203138, -0.17192985)]

    result_df = pd.DataFrame(result_tuple, columns=['Player_ID', 'Score'])

    focus_list = []
    size = result_df.shape[0]
    for index, row in result_df.iterrows():
        p = int((1-float(index+1)/float(size+1))*100.0)
        if p>=50:
            focus_list.append(row['Player_ID'])



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


    df_id_result = {}
    df_n_result = {} 
    df_e_result = {}
    df_result = {}
    df_sorted = {}
    df_top3 = {}

    rookie_id = []
    first_degree = []
    second_degree = []
    third_degree = []
    first_degree_name = []
    second_degree_name = []
    third_degree_name = []



    count = 0
    save = {}
    for name, g in grouped['2013']:

        count = count+1

        n = g.shape[0]
        m = g.mean()
        #print name, n, m

        if m['Player_ID'] not in focus_list:
            continue

        print m['Player_ID']

        bkg_unit = np.ravel(g.as_matrix(columns=['PTS','AST','REB','STL','BLK','FG_PCT','FG3_PCT','FT_PCT','MIN','EFF','WL']))
     
        #print bkg_unit.shape
 
        bkg_container = []
        bkg_container.append(bkg_unit)
        #print bkg_container

        sig_str = 'sig_%dg' % n

        cur[sig_str]  = db.cursor(mdb.cursors.DictCursor)
        cur[sig_str].execute("SELECT PTS,AST,REB,STL,BLK,FGA,FGM,FTA,FTM,TOV,WL,FG_PCT,FG3_PCT,FT_PCT,MIN,FROM_YEAR,Player_ID FROM star WHERE PLAYED<=%d;" % n)
        df[sig_str] = pd.DataFrame( cur[sig_str].fetchall() )

        # calculating effciency
        df[sig_str]['EFF'] = (df[sig_str]['PTS'] + df[sig_str]['REB'] + df[sig_str]['AST'] + df[sig_str]['STL'] + df[sig_str]['BLK']) - ((df[sig_str]['FGA']-df[sig_str]['FGM'])+(df[sig_str]['FTA']-df[sig_str]['FTM'])+df[sig_str]['TOV'])

        df[sig_str] = df[sig_str][ df[sig_str]['FROM_YEAR'] < 2013 ]

        # trim down variables
        df[sig_str] = df[sig_str].ix[:,['PTS','AST','REB','STL','BLK','FG_PCT','FG3_PCT','FT_PCT','MIN','EFF','WL','Player_ID']]

        df[sig_str]['WL'] = df[sig_str]['WL'].map(lambda x: 1 if x=='W' else 0)

        sig_container = []
        sig_id_container = []
        sig_name_container = []

        sgrouped = df[sig_str].groupby('Player_ID')
        for sname, sg in sgrouped:
            #print sg.shape[0],n
            if sg.shape[0] != n:
                continue
            sm = sg.mean()
            print sm['Player_ID']
            sig_unit = np.ravel(sg.as_matrix(columns=['PTS','AST','REB','STL','BLK','FG_PCT','FG3_PCT','FT_PCT','MIN','EFF','WL']))
            #print sig_unit.shape
            sig_container.append(sig_unit)
            sig_id_container.append(sm['Player_ID'])

            for index2, row2 in all_players.iterrows():
                if int(row2["PERSON_ID"]) == int(sm['Player_ID']):
                    sig_name_container.append(row2['DISPLAY_LAST_COMMA_FIRST'])




        #print sig_container

        rookie = str(int(m['Player_ID']))

        df_id_result[rookie] = pd.DataFrame(sig_id_container, columns=['StarID'])
        df_n_result[rookie] = pd.DataFrame(sig_name_container, columns=['StarName'])
        e_result = euclidean_distances(sig_container, bkg_container)
        df_e_result[rookie] = pd.DataFrame(e_result, columns=['Distance'])
        #print df_n_result[rookie]
        #print df_e_result[rookie]
        df_result[rookie] = df_id_result[rookie].join([df_n_result[rookie], df_e_result[rookie]])
        #print df_result[rookie]

        df_sorted[rookie] = df_result[rookie].sort(['Distance'], ascending=1) 
        df_top3[rookie] = df_sorted[rookie][:3]
         
        print df_top3[rookie]

        c = 0
        for i, r in df_top3[rookie].iterrows():
            if c==0:
                rookie_id.append(rookie)
                first_degree.append(r['StarID'])
                first_degree_name.append(r['StarName'])
            elif c==1:
                second_degree.append(r['StarID'])
                second_degree_name.append(r['StarName'])
            elif c==2:
                third_degree.append(r['StarID'])
                third_degree_name.append(r['StarName'])
            c = c+1

    a = pd.DataFrame(rookie_id, columns=['RookieID'])
    b = pd.DataFrame(first_degree, columns=['FirstDegreeID'])
    c = pd.DataFrame(second_degree, columns=['SecondDegreeID'])
    d = pd.DataFrame(third_degree, columns=['ThirdDegreeID'])
    x = pd.DataFrame(first_degree_name, columns=['FirstDegreeName'])
    y = pd.DataFrame(second_degree_name, columns=['SecondDegreeName'])
    z = pd.DataFrame(third_degree_name, columns=['ThirdDegreeName'])
    f = a.join([b, c, d, x, y, z])
    print f

    f.to_csv('../log/0923_similarity.csv', sep='\t', index=False)



if __name__ == '__main__':
    main()

