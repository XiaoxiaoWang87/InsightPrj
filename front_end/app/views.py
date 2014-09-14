#!/usr/bin/python

from flask import render_template, jsonify, request
from app import app
import pymysql as mdb

import random

import pandas as pd
import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn_pandas import DataFrameMapper, cross_val_score

from patsy import dmatrices

db = mdb.connect(user="root", host="localhost", db="gamelogdb", charset='utf8')  #db="world_innodb", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
        title = 'Home', user = { 'nickname': 'Miguel' },
        )


@app.route('/regression')
def regression():
    """Add two numbers server side, ridiculous but well..."""
    pts = request.args.get('a', 0, type=float)
    ast = request.args.get('b', 0, type=float)
    reb = request.args.get('c', 0, type=float)
    n = request.args.get('d', 0, type=float)
    #n = 100 #request.args.get('NGAMES', 0, type=float)

    df = {}

    cur = {}

    prob = []
    train_s = []
    train_o = []
    mis_r = []

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

    #df[sig_str+'_testing'] = df[sig_str].drop(train_sig[sig_str])
    #df[bkg_str+'_intermediate'] = df[bkg_str].drop(train_bkg[bkg_str])
    #df[bkg_str+'_testing'] = df[bkg_str].ix[ random.sample(df[bkg_str+'_intermediate'].index, int(df[sig_str+'_testing'].shape[0])) ]

    df_sig_train = df[sig_str+'_training']
    df_bkg_train = df[bkg_str+'_training'] 

    X_sig = np.array(df_sig_train)
    y_sig = np.array(X_sig.shape[0] * [1])
    
    X_bkg = np.array(df_bkg_train)
    y_bkg = np.array(X_bkg.shape[0] * [0])


    X = np.concatenate((X_sig, X_bkg))
    y = np.concatenate((y_sig, y_bkg))

    model = LogisticRegression()

    model.fit(X, y)

    i=[float(pts), float(ast), float(reb)]
    input_n = np.array(i)
    prob = round(((model.predict_proba(input_n))[0][1])*100,0)
    conf = round((model.score(X, y))*100,0)
    return jsonify(result=[prob, conf])





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


