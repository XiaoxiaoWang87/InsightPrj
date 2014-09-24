import pandas as pd
import numpy as np
import json

import requests
import pandas as pd


with open("../json/names.json", 'r') as file_handle:
    names = json.load(file_handle)
    #print names
    all_players = pd.DataFrame(names['resultSets'][0]['rowSet'], columns=names['resultSets'][0]['headers'])

#all_players = all_players[all_players['FROM_YEAR']=='2013']

#all_players["URL"] = all_players['PERSON_ID'].map(lambda x: 'http://stats.nba.com/media/players/230x185/'+str(x)+'.png')

#all_players.to_csv('../log/0914_prediction_profile.csv', sep='\t', index=False)

#for index, row in all_players.iterrows():
#    print '<option value='+row['DISPLAY_LAST_COMMA_FIRST']+'>'

all_players = all_players[["PERSON_ID", "FROM_YEAR"]]
all_players.columns = ['Player_ID', 'FROM_YEAR']

df = {}
df['sig'] = pd.read_csv('../log/0914_allstar_post1980_sql_log.csv',sep='\t')
df['bkg'] = pd.read_csv('../log/0914_nonstar_post1980_sql_log.csv',sep='\t')

df['sig'] = pd.merge(df['sig'], all_players, on='Player_ID', how='left')
df['bkg'] = pd.merge(df['bkg'], all_players, on='Player_ID', how='left')

df['sig'].to_csv('../log/0918_allstar_post1980_sql_log.csv',sep='\t')
df['bkg'].to_csv('../log/0918_nonstar_post1980_sql_log.csv',sep='\t')

