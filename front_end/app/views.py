#!/usr/bin/python

from flask import render_template, jsonify, request
from app import app
import pymysql as mdb

import random

import pandas as pd
import numpy as np

#import sklearn
#from sklearn import preprocessing
#from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
#from sklearn import metrics
#from sklearn.cross_validation import cross_val_score
#from sklearn.feature_extraction import DictVectorizer
#from sklearn_pandas import DataFrameMapper, cross_val_score
#
#from patsy import dmatrices

#db = mdb.connect(user="root", host="localhost", db="gamelogdb", charset='utf8')  #db="world_innodb", charset='utf8')

db = mdb.connect(user="root", host="localhost", db="nbagamedb", charset='utf8')
#db = mdb.connect(user="root", host="localhost", db="demodb", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index_final.html",
        title = 'Home', user = { 'nickname': 'Miguel' },
        )


@app.route('/regression')
def regression():
    """Add two numbers server side, ridiculous but well..."""
    p_name = request.args.get('d', 0, type=str)
    print p_name  

    d1 = ''
    d2 = ''
    d3 = ''

    n1 = ''
    n2 = ''
    n3 = ''

    df = {}
    cur = {}
    p_profile = "player_profile"
    cur[p_profile]  = db.cursor(mdb.cursors.DictCursor)
    cur[p_profile].execute("SELECT FULLNAME,URL,PERSONID FROM rookie_profile;")
    df[p_profile] = pd.DataFrame( cur[p_profile].fetchall() )

    #print df[p_profile]

    url = "no"
    pid = "0"
    p = -1
    for i, row in df[p_profile].iterrows():
        if p_name == row["FULLNAME"]:
            url = row["URL"]
            pid = row["PERSONID"]


    height = ""
    weight = ""
    born = ""
    p_bib = "player_bibliography"
    cur[p_bib]  = db.cursor(mdb.cursors.DictCursor)
    cur[p_bib].execute("SELECT PERSON_ID,PLAYERCODE,HEIGHT,WEIGHT,BORN FROM bibliography;")
    df[p_bib] = pd.DataFrame( cur[p_bib].fetchall() )

    for i, row in df[p_bib].iterrows():
        if pid == row["PERSON_ID"]:
            height = row["HEIGHT"]
            weight = row["WEIGHT"]
            born = row["BORN"]

    p_frac = "player_fraction"
    cur[p_frac] = db.cursor(mdb.cursors.DictCursor)
    cur[p_frac].execute("SELECT Player_ID,Score FROM score;")
    df[p_frac] = pd.DataFrame( cur[p_frac].fetchall() )
    
    size = df[p_frac].shape[0]
    for index, row in df[p_frac].iterrows():
        if pid == row["Player_ID"]:
            p = int((1-float(index+1)/float(size+1))*100.0)


    p_sim = "player_similarity"
    cur[p_sim] = db.cursor(mdb.cursors.DictCursor)
    cur[p_sim].execute("SELECT RookieID,FirstDegreeID,SecondDegreeID,ThirdDegreeID,FirstDegreeName,SecondDegreeName,ThirdDegreeName FROM similarity;")
    df[p_sim] = pd.DataFrame( cur[p_sim].fetchall() )
    for index, row in df[p_sim].iterrows():
        if pid == row["RookieID"]:
            d1 = "http://stats.nba.com/media/players/230x185/"+str(int(row["FirstDegreeID"]))+".png"
            d2 = "http://stats.nba.com/media/players/230x185/"+str(int(row["SecondDegreeID"]))+".png"
            d3 = "http://stats.nba.com/media/players/230x185/"+str(int(row["ThirdDegreeID"]))+".png"
            n1 = row["FirstDegreeName"] 
            n2 = row["SecondDegreeName"]
            n3 = row["ThirdDegreeName"]


    #####pts = request.args.get('a', 0, type=float)
    #####ast = request.args.get('b', 0, type=float)
    #####reb = request.args.get('c', 0, type=float)
    #####n = request.args.get('d', 0, type=float)
    ######n = 100 #request.args.get('NGAMES', 0, type=float)

    #####df = {}

    #####cur = {}

    #####prob = []
    #####train_s = []
    #####train_o = []
    #####mis_r = []

    #####sig_str = 'sig_%dg' % n
    #####bkg_str = 'bkg_%dg' % n

    #####cur[sig_str]  = db.cursor(mdb.cursors.DictCursor)
    #####cur[sig_str].execute("SELECT PTS,AST,REB FROM star WHERE PLAYED<=%d AND LEARN_INDICATOR=1;" % n)
    #####df[sig_str] = pd.DataFrame( cur[sig_str].fetchall() )

    #####cur[bkg_str]  = db.cursor(mdb.cursors.DictCursor)
    #####cur[bkg_str].execute("SELECT PTS,AST,REB FROM non_star WHERE PLAYED<=%d AND LEARN_INDICATOR=1;" % n)
    #####df[bkg_str] = pd.DataFrame( cur[bkg_str].fetchall() )


    #####train_sig = {}
    #####train_bkg = {}
    ###### Signal: use 80% for training, 20% for testing
    ###### Background: use sig*5 for training, rest for testing 
    #####train_sig[sig_str] = random.sample(df[sig_str].index, int(df[sig_str].shape[0] * 0.6)) #int(df['sig_10g'].shape[0] * 0.8))
    #####train_bkg[bkg_str] = random.sample(df[bkg_str].index, len(train_sig[sig_str]))#int(df[bkg_str].shape[0] * 0.6)) #int(df['sig_10g'].shape[0] * 0.8 * 5))

    #####df[sig_str+'_training'] = df[sig_str].ix[ train_sig[sig_str] ]
    #####df[bkg_str+'_training'] = df[bkg_str].ix[ train_bkg[bkg_str] ]

    ######df[sig_str+'_testing'] = df[sig_str].drop(train_sig[sig_str])
    ######df[bkg_str+'_intermediate'] = df[bkg_str].drop(train_bkg[bkg_str])
    ######df[bkg_str+'_testing'] = df[bkg_str].ix[ random.sample(df[bkg_str+'_intermediate'].index, int(df[sig_str+'_testing'].shape[0])) ]

    #####df_sig_train = df[sig_str+'_training']
    #####df_bkg_train = df[bkg_str+'_training'] 

    #####X_sig = np.array(df_sig_train)
    #####y_sig = np.array(X_sig.shape[0] * [1])
    #####
    #####X_bkg = np.array(df_bkg_train)
    #####y_bkg = np.array(X_bkg.shape[0] * [0])


    #####X = np.concatenate((X_sig, X_bkg))
    #####y = np.concatenate((y_sig, y_bkg))

    #####model = LogisticRegression()

    #####model.fit(X, y)

    #####i=[float(pts), float(ast), float(reb)]
    #####input_n = np.array(i)
    #####prob = round(((model.predict_proba(input_n))[0][1])*100,0)
    #####conf = round((model.score(X, y))*100,0)
    #return jsonify(result=[prob, conf])
    ploc = {}
    #pid = "static/Gaussian/Gaussian_"+str(pid)+".jpg"
    ploc['PTS'] = "static/images/Performance/PTS_"+str(pid)+".jpg"
    ploc['AST'] = "static/images/Performance/AST_"+str(pid)+".jpg"
    ploc['REB'] = "static/images/Performance/REB_"+str(pid)+".jpg"
    ploc['MIN'] = "static/images/Performance/MIN_"+str(pid)+".jpg"
    ploc['FG_PCT'] = "static/images/Performance/FG_PCT_"+str(pid)+".jpg"
    ploc['EFF'] = "static/images/Performance/EFF_"+str(pid)+".jpg"
    print url,pid,p,n1,n2,n3
    return jsonify(result=[url,p,ploc['PTS'], ploc['AST'], ploc['REB'], ploc['MIN'], ploc['FG_PCT'], ploc['EFF'], d1, d2, d3, n1, n2, n3, height, weight, born])




@app.route('/db')
def cities_page():
    with db: 
        cur = db.cursor()
        cur.execute("SELECT Name FROM city LIMIT 15;")
        query_results = cur.fetchall()
    cities = ""
    for result in query_results:
        cities += result[0]
        cities += "<br>"
    return cities

@app.route("/db_fancy")
def cities_page_fancy():
    with db:
        cur = db.cursor()
        cur.execute("SELECT Name, CountryCode, Population FROM City ORDER BY Population LIMIT 15;")

        query_results = cur.fetchall()
    cities = []
    for result in query_results:
        cities.append(dict(name=result[0], country=result[1], population=result[2]))
    return render_template('cities.html', cities=cities) 


@app.route("/db_json")
def cities_json():
    with db:
        cur = db.cursor()
        #cur.execute("SELECT Name, CountryCode, Population FROM city ORDER BY Population DESC;")
        cur.execute("SELECT * FROM star;")
        query_results = cur.fetchall()
    cities = []
    for result in query_results:
        cities.append(dict(name=result[0], country=result[1], population=result[2]))
    return jsonify(dict(cities=cities))


@app.route("/jquery")
def index_jquery():
    return render_template('index_js.html') 


