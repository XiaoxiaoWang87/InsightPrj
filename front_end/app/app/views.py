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

#db = mdb.connect(user="root", host="localhost", db="demodb", charset='utf8')

#db = mdb.connect(user="root", host="localhost", db="nbagamedb", charset='utf8')

@app.route('/')
@app.route('/index')
def index():
    return render_template("index_final.html",
        title = 'Home', user = { 'nickname': 'Miguel' },
        )


@app.route('/regression')
def regression():

    db = mdb.connect(user="root", host="localhost", db="nbagamedb", charset='utf8')

    """Add two numbers server side, ridiculous but well..."""
    p_name = request.args.get('d', 0, type=str)
    print p_name  

    df = {}
    cur = {}


    url = "no"
    pid = "0"

    # Get URL link and player id
    p_profile = "player_profile"
    print "test"
    cur[p_profile]  = db.cursor(mdb.cursors.DictCursor)
    cur[p_profile].execute("SELECT FULLNAME,URL,PERSONID FROM rookie_profile;")
    df[p_profile] = pd.DataFrame( cur[p_profile].fetchall() )

    #for i, row in df[p_profile].iterrows():
    #    if p_name == row["FULLNAME"]:
    #        url = row["URL"]
    #        pid = row["PERSONID"]

    if p_name in (df[p_profile]["FULLNAME"]).tolist():
        url = ( df[p_profile]['URL'][df[p_profile]['FULLNAME'] == p_name] ).values[0]
        pid = ( df[p_profile]['PERSONID'][df[p_profile]['FULLNAME'] == p_name] ).values[0]


    height = ""
    weight = ""
    born = ""

    # Get player bibliography
    p_bib = "player_bibliography"
    cur[p_bib]  = db.cursor(mdb.cursors.DictCursor)
    cur[p_bib].execute("SELECT PERSON_ID,PLAYERCODE,HEIGHT,WEIGHT,BORN FROM bibliography;")
    df[p_bib] = pd.DataFrame( cur[p_bib].fetchall() )

    #for i, row in df[p_bib].iterrows():
    #    if pid == row["PERSON_ID"]:
    #        height = row["HEIGHT"]
    #        weight = row["WEIGHT"]
    #        born = row["BORN"]
 
    if pid in (df[p_bib]["PERSON_ID"]).tolist():
        height = ( df[p_bib]['HEIGHT'][df[p_bib]['PERSON_ID'] == pid] ).values[0]
        weight = ( df[p_bib]['WEIGHT'][df[p_bib]['PERSON_ID'] == pid] ).values[0]
        born = ( df[p_bib]['BORN'][df[p_bib]['PERSON_ID'] == pid] ).values[0]


    prob = -1
    # Get player likelihood
    p_frac = "player_fraction"
    cur[p_frac] = db.cursor(mdb.cursors.DictCursor)
    cur[p_frac].execute("SELECT Player_ID,Score FROM score;")
    df[p_frac] = pd.DataFrame( cur[p_frac].fetchall() )
    
    size = df[p_frac].shape[0]
    for index, row in df[p_frac].iterrows():
        if pid == row["Player_ID"]:
            prob = int((1-float(index+1)/float(size+1))*100.0)


    d1 = ''
    d2 = ''
    d3 = ''

    n1 = ''
    n2 = ''
    n3 = ''

    # Get player similarity
    p_sim = "player_similarity"
    cur[p_sim] = db.cursor(mdb.cursors.DictCursor)
    cur[p_sim].execute("SELECT RookieID,FirstDegreeID,SecondDegreeID,ThirdDegreeID,FirstDegreeName,SecondDegreeName,ThirdDegreeName FROM similarity;")
    df[p_sim] = pd.DataFrame( cur[p_sim].fetchall() )
    #for index, row in df[p_sim].iterrows():
    #    if pid == row["RookieID"]:
    #        d1 = "http://stats.nba.com/media/players/230x185/"+str(int(row["FirstDegreeID"]))+".png"
    #        d2 = "http://stats.nba.com/media/players/230x185/"+str(int(row["SecondDegreeID"]))+".png"
    #        d3 = "http://stats.nba.com/media/players/230x185/"+str(int(row["ThirdDegreeID"]))+".png"
    #        n1 = row["FirstDegreeName"] 
    #        n2 = row["SecondDegreeName"]
    #        n3 = row["ThirdDegreeName"]
    if pid in (df[p_sim]["RookieID"]).tolist():
        d1 = "http://stats.nba.com/media/players/230x185/"+str(int( (df[p_sim]['FirstDegreeID'][df[p_sim]['RookieID'] == pid]).values[0] ))+".png"
        d2 = "http://stats.nba.com/media/players/230x185/"+str(int( (df[p_sim]["SecondDegreeID"][df[p_sim]['RookieID'] == pid]).values[0] ))+".png"
        d3 = "http://stats.nba.com/media/players/230x185/"+str(int( (df[p_sim]["ThirdDegreeID"][df[p_sim]['RookieID'] == pid]).values[0] ))+".png"
        n1 = (df[p_sim]["FirstDegreeName"][df[p_sim]['RookieID'] == pid]).values[0]
        n2 = (df[p_sim]["SecondDegreeName"][df[p_sim]['RookieID'] == pid]).values[0]
        n3 = (df[p_sim]["ThirdDegreeName"][df[p_sim]['RookieID'] == pid]).values[0]
        

    ploc = {}
    # Get performance
    #pid = "static/Gaussian/Gaussian_"+str(pid)+".jpg"
    if pid in (df[p_profile]["PERSONID"]).tolist():
        ploc['PTS'] = "static/images/Performance/PTS_"+str(pid)+".jpg"
        ploc['AST'] = "static/images/Performance/AST_"+str(pid)+".jpg"
        ploc['REB'] = "static/images/Performance/REB_"+str(pid)+".jpg"
        ploc['MIN'] = "static/images/Performance/MIN_"+str(pid)+".jpg"
        ploc['FG_PCT'] = "static/images/Performance/FG_PCT_"+str(pid)+".jpg"
        ploc['EFF'] = "static/images/Performance/EFF_"+str(pid)+".jpg"
    else:
        ploc['PTS'] = ""
        ploc['AST'] = ""
        ploc['REB'] = ""
        ploc['MIN'] = ""
        ploc['FG_PCT'] = ""
        ploc['EFF'] = ""

    print url,pid,prob,n1,n2,n3
    return jsonify(result=[url,prob,ploc['PTS'], ploc['AST'], ploc['REB'], ploc['MIN'], ploc['FG_PCT'], ploc['EFF'], d1, d2, d3, n1, n2, n3, height, weight, born])




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


