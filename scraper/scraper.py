import pandas as pd
import json

import requests
from bs4 import BeautifulSoup
import urllib2

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver

url = 'http://stats.nba.com/playerProfile.html?PlayerID='

player_id = []
#player_id = ['78602','893']

with open("../json/names.json", 'r') as file_handle:
    names = json.load(file_handle)
    all_players = pd.DataFrame(names['resultSets'][0]['rowSet'], columns=names['resultSets'][0]['headers'])
    i = 0
    for index, row in all_players.iterrows():
        #if i>=0:
        if float(row["FROM_YEAR"]) < 1980:
            continue
        player_id.append((row['PLAYERCODE'], row['PERSON_ID']))
        print row['PLAYERCODE'], row['PERSON_ID']

        i = i+1

wd = webdriver.Firefox()


position_l = []
draft_l = []

df = pd.DataFrame(columns=['PERSON_ID','PLAYERCODE','POSITION','DRAFT'], index= list(xrange(len(player_id))))

count=0
for i in player_id:
    print i[0], i[1]
    wd.get(url + str(i[1]))
    #WebDriverWait(wd, 10).until(
    #  EC.visibility_of_element_located((By.CLASS_NAME, "carousel")))
    html_page = wd.page_source
    
    #page=urllib2.urlopen(url+player_id)
    #soup = BeautifulSoup(page.read())
    soup = BeautifulSoup(html_page, 'html.parser')
    
    position=soup.findAll('h2',{'class':'num-position'})
    draft = soup.findAll('div',{'class':'detail-text', 'id':'draft'})
    

    for p in position:
        position_l.append(p.contents)
        print p.contents
    
    for d in draft:
        draft_l.append(d.contents)
        print d.contents

    df.loc[count] = pd.Series({'PERSON_ID':i[1], 'PLAYERCODE':i[0], 'POSITION':p.contents, 'DRAFT':d.contents})

    count = count + 1

wd.quit()

print df
df.to_csv('position_draft.csv', sep='\t')



