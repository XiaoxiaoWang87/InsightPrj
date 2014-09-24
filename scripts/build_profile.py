

import pandas as pd
import numpy as np
import json

import requests
import pandas as pd


with open("../json/names.json", 'r') as file_handle:
    names = json.load(file_handle)
    #print names
    all_players = pd.DataFrame(names['resultSets'][0]['rowSet'], columns=names['resultSets'][0]['headers'])

all_players = all_players[all_players['FROM_YEAR']=='2013']

all_players["URL"] = all_players['PERSON_ID'].map(lambda x: 'http://stats.nba.com/media/players/230x185/'+str(x)+'.png') 

all_players.to_csv('../log/0914_prediction_profile.csv', sep='\t', index=False)

for index, row in all_players.iterrows():
    print '<option value='+row['DISPLAY_LAST_COMMA_FIRST']+'>'
