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

import pymysql as mdb

con = mdb.connect('localhost', 'root', '', 'gamelogdb')

df = {}

df['sig'] = pd.read_csv('../log/allstar_post1980_sql_log.csv',sep='\t')
df['bkg'] = pd.read_csv('../log/nonstar_post1980_sql_log.csv',sep='\t')

# features: a, b, c ...
df['sig'].to_sql('star', con, 'mysql', 'replace')  #,'fail',True)
df['bkg'].to_sql('non_star', con, 'mysql', 'replace')
