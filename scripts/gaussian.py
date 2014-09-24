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


db = mdb.connect(user="root", host="localhost", db="nbagamedb", charset='utf8')

cur = db.cursor(mdb.cursors.DictCursor)
cur.execute("SELECT Player_ID,Score FROM score;")
df = pd.DataFrame( cur.fetchall() )

print df
size = df.shape[0]
print size
#special.erfinv((0.5*2-1))*sqrt(2)
for index, row in df.iterrows():

    #if index!=0:
    #    continue

    p = float(index+1)/float(size+1)
    x_bm = special.erfinv(( (1-p)*2-1))*math.sqrt(2)

    mean = 0
    variance = 1
    sigma = np.sqrt(variance)
    x = np.linspace(-3,3,100)
    y = mlab.normpdf(x,mean,sigma)

    mpl.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.text(-2.5, 0.41, 'Relative Chance of Becoming NBA All-star Player: %s%%' % (100- int(p*100)), fontsize=14)
    plt.title('Low <--------> Med <--------> High')
    plt.plot(x,y,'r',lw=2)
    ax.fill(x,y, 'r',alpha=.3)
    ax.set_xlabel('Player Relative Performance')
    ax.set_ylabel('Arbituary Unit')
    plt.grid(True)
    plt.plot([x_bm, x_bm], [0, 0.4], color='grey', linewidth=2)
    plt.savefig("Gaussian_"+str(int(row['Player_ID']))+".jpg")
    plt.close()
    #plt.show()
