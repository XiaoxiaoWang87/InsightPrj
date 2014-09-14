#http://stats.nba.com/stats/commonteamroster/?Season=2001-02&LeagueID=00&TeamID=1610616833&SeasonType=All+Star

import pandas as pd
import numpy as np
import json

import requests

#names = pd.read_json("names.json")

#pd.set_option('notebook_repr_html',True)
#pd.set_option('display.max_columns',300)
#pd.set_option('display.width',3000)

class AllStarLog:
    def __init__(self, teamid, season='2013-14',seasontype='All+Star', leagueid='00'):
        self._url = "http://stats.nba.com/stats/commonteamroster/?"
        self._api_param = {
                          'Season': season,
                          'LeagueID': leagueid,
                          'TeamID':teamid,
                          'SeasonType': seasontype,
                          }
        self._x = requests.get(self._url, params=self._api_param)
        self._x = self._x.json()
    def check(self):
        return len(self._x['resultSets'][0]['rowSet'])
    def log(self):
        return pd.DataFrame(self._x['resultSets'][0]['rowSet'],columns=self._x['resultSets'][0]['headers'])

allstar_log={}
index = 0

for i in range(1950, 2014):
    year = str(i) + '-' + str(i+1)[2:]
    print year
    
    allstar_log['east'] = AllStarLog(1610616833, year, 'All+Star')
    allstar_log['west'] = AllStarLog(1610616834, year, 'All+Star')

    if allstar_log['east'].check() == 0 or allstar_log['west'].check() == 0:
        continue

    east_df = allstar_log['east'].log()
    west_df = allstar_log['west'].log()

    if index == 0:
        east_df.to_csv('allstar_log.csv', sep='\t', index=False)
        west_df.to_csv('allstar_log.csv', mode='a', sep='\t', header=False, index=False)
    else:
        east_df.to_csv('allstar_log.csv', mode='a', sep='\t', header=False, index=False)
        west_df.to_csv('allstar_log.csv', mode='a', sep='\t', header=False, index=False)

    index = index + 1

#with open("names.json", 'r') as file_handle:
#    names = json.load(file_handle)
#    #print names
#    all_players = pd.DataFrame(names['resultSets'][0]['rowSet'], columns=names['resultSets'][0]['headers'])
#
#    for index, row in all_players.iterrows():
#        if index < 20:
#            #print row.get('PERSON_ID', np.nan)
#            print row['PLAYERCODE']
#            player_gamelog = AllStarLog(row['PERSON_ID'], 'ALL', 'Regular Season')
#            if player_gamelog.check() == 0:
#                continue
#            #print player_gamelog.log()
#            player_gamelog_df = player_gamelog.log()
#            if index == 0: 
#                player_gamelog_df.to_csv('player_gamelog.csv', sep='\t', index=False)
#            else:
#                player_gamelog_df.to_csv('player_gamelog.csv', mode='a', sep='\t', header=False, index=False)
