
# get all players (since 1980) game log data
# regular season / playoffs

import pandas as pd
import numpy as np
import json

import requests
import pandas as pd

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


with open("../json/names.json", 'r') as file_handle:
    names = json.load(file_handle)
    #print names
    all_players = pd.DataFrame(names['resultSets'][0]['rowSet'], columns=names['resultSets'][0]['headers'])

    for index, row in all_players.iterrows():
        #if index > 20:

        if float(row["FROM_YEAR"]) < 1980:
            continue
        #print row.get('PERSON_ID', np.nan)
        print row['PLAYERCODE']
        player_gamelog = GameLog(row['PERSON_ID'], 'ALL', 'Regular Season')
        #player_gamelog = GameLog(row['PERSON_ID'], 'ALL', 'Playoffs')
        if player_gamelog.check() == 0:
            continue
        #print player_gamelog.log()
        player_gamelog_df = player_gamelog.log()
        if index == 0: 
            #player_gamelog_df.to_csv('player_gamelog.csv', sep='\t', index=False)
            #player_gamelog_df.to_csv('../log/player_playoffs_gamelog.csv', sep='\t', index=False)
            player_gamelog_df.to_csv('../log/0914_player_gamelog.csv', sep='\t', index=False)
            #player_gamelog_df.to_csv('../log/0914_player_playoffs_gamelog.csv', sep='\t', index=False)
        else:
            #player_gamelog_df.to_csv('../log/player_playoffs_gamelog.csv', mode='a', sep='\t', header=False, index=False)
            player_gamelog_df.to_csv('../log/0914_player_gamelog.csv', mode='a', sep='\t', header=False, index=False)
            #player_gamelog_df.to_csv('../log/0914_player_playoffs_gamelog.csv', mode='a', sep='\t', header=False, index=False)


