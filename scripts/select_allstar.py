import pandas as pd
import numpy as np
import json

import requests

#names = pd.read_json("names.json")

#pd.set_option('notebook_repr_html',True)
#pd.set_option('display.max_columns',300)
#pd.set_option('display.width',3000)

class GameLog:
    def __init__(self, playerid, season='ALL',seasontype='Regular Season', leagueid=''):
        self._url = "http://stats.nba.com/stats/playergamelog?"
        self._api_param = {'PlayerID':playerid,
                          'SeasonType': seasontype,
                          'Season': season,
                          'LeagueID': leagueid
                          }
        self._x = requests.get(self._url, params=self._api_param)
        self._x = self._x.json()
    def check(self):
        return len(self._x['resultSets'][0]['rowSet'])
    def log(self):
        return pd.DataFrame(self._x['resultSets'][0]['rowSet'],columns=self._x['resultSets'][0]['headers'])


allstar_list = []
allstar_df = pd.read_csv('allstar_log.csv',sep='\t')
 
counter = 0
for index, row in allstar_df.iterrows():


    if row['PLAYER_ID'] not in allstar_list:
        allstar_list.append(row['PLAYER_ID'])
    else:
        continue

    if row['SEASON'] < 1980:
        continue

    print row['PLAYER']

    player_gamelog = GameLog(row['PLAYER_ID'], 'ALL', 'Regular Season')

    if player_gamelog.check() == 0:
        continue

    allstar_gamelog_df = player_gamelog.log()

    if counter == 0: 
        allstar_gamelog_df.to_csv('allstar_gamelog.csv', sep='\t', index=False)
    else:
        allstar_gamelog_df.to_csv('allstar_gamelog.csv', mode='a', sep='\t', header=False, index=False)

    counter = counter+1
