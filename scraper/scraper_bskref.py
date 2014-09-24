import re
import pandas as pd
import json

import requests
from bs4 import BeautifulSoup
import urllib2

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver


df = pd.read_csv('position_draft.csv',sep='\t')

pos = []
llast = []
lf5_last = []
lf1_last = []
lfirst = []
lf2_first = []

player_id = []

for index, row in df.iterrows():
    #if row["POSITION"] != '[]':
    #    POS = row["POSITION"].split('|')[1][1:].replace("']", "")
    #    if POS in pos:
    #        continue
    #    else:
    #        pos.append(POS)
    print row["PLAYERCODE"]
    player_id.append((row['PLAYERCODE'], row['PERSON_ID']))

    first_last = str(row["PLAYERCODE"]).lower().split("_")

    if first_last[0] == 'histadd':
        last = first_last[2]
        f5_last = last[:5]
        f1_last = last[:1]
        first = first_last[1] 
        f2_first = first[:2]  
    else:
        last = first_last[1]
        f5_last = last[:5]
        f1_last = last[:1]
        first = first_last[0]
        f2_first = first[:2]
    
    print f5_last
    llast.append(last)
    lf5_last.append(f5_last)
    lf1_last.append(f1_last)
    lfirst.append(first)
    lf2_first.append(f2_first)


#wd = webdriver.Firefox()
position_l = []
draft_l = []
url="http://www.basketball-reference.com/players/"


df = pd.DataFrame(columns=['PERSON_ID','PLAYERCODE','POSITION','DRAFT'], index= list(xrange(len(player_id))))


for i in range(len(llast)):
 
    if i>=5:
        continue

    link = url + lf1_last[i] + '/' + lf5_last[i] + lf2_first[i] + '01.html'
    print link
    #link = 'http://www.basketball-reference.com/players/a/abdursh01.html'

    page=urllib2.urlopen(link)
    #wd.get(link)
 
    #html_page = wd.page_source
    #soup = BeautifulSoup(html_page, 'html.parser')
    soup = BeautifulSoup(page.read())
    

    #person_images = soup.findAll('div',{'class':'person_image_offset'})
    #person_images = soup.findAll('div',{'id':'info_box'})

    #for p in person_images:
    #    print p.contents

    
    person_images = soup.findAll('p')

    for p in person_images:
        #print p.contents
        print len(p.contents)
        
        #if c>10:
        #    continue

    #    has_pos = False
    #    which = 7
    #    for s in range(len(p.contents)):
    #        if "Position:" in p.contents[s].encode('utf-8').strip():
    #            print "Found!"
    #            which = s
    #            has_pos = True

    #    if has_pos == True:
    #        shortest = p.contents[which+1].encode('utf-8').strip()
    #        print shortest
    #    else:
    #        shortest = ""

    #    position_l.append(shortest)
        #print shortest
        #position_l.append(shortest)
        #if has_pos == True:
        #    shorter = (str(p.contents[which]).split("</span>")[1]).split("&") #.contents[9]
        #    shortest = shorter[0].split('<span')[0]
        #else:
        #    shortest = ""
        #print shortest
        #position_l.append(shortest)
        #c = c+1

print position_l

    #c = 0

    #for p in person_images:

    #    if c>0:
    #        continue
    #    print p.contents


    #    has_pos = False
    #    which = 7
    #    for s in range(len(p.contents)):

    #        if "Position:" in str(p.contents[s]):
    #            which = s
    #            has_pos = True

    #    if has_pos == True:
    #        shorter = (str(p.contents[which]).split("</span>")[1]).split("&") #.contents[9]
    #        #cleaner = shorter[0].encode('utf-8')
    #        #print cleaner
    #        shortest = shorter[0].split('<span')[0]
    #    else:
    #        shortest = ""
    #    print shortest
    #    position_l.append(shortest)


    #    has_draft = False
    #    which2 = 8
    #    for s in range(len(p.contents)):
    #        if "draft" in str(p.contents[s]):
    #            which2 = s
    #            has_draft = True
    #    if has_draft == True:
    #        shorter2 = (str(p.contents[which2]).split("draft")[1]).split(",")
    #    #if len(shorter2)>=3:
    #        shortest2 = shorter2[1]+shorter2[2]
    #    else:
    #        shortest2 = ""

    #    #elif len(shorter2)>=2:
    #    #    shortest2 = shorter2[1]
    #    #else:
    #    #    shortest2 =''
    #    print shortest2
    #    draft_l.append(shortest2)

    #    df.loc[c] = pd.Series({'PERSON_ID':player_id[i][1], 'PLAYERCODE':player_id[i][0], 'POSITION':shortest, 'DRAFT':shortest2})

    #    c = c+1


    #print df
    #df.to_csv('position_draft_new.csv', sep='\t')

        #r = re.compile('</a>, (.*?), <a href="/draft')
        #print type(shorter2[0])
        #m = re.search('</a>, (.*?), <a href="/draft', str(shorter2[0]))
        #if m:
        #    lyrics = m.group(1)
        #    print lyrics
        #r = re.compile(r'(\d+\.\d+)')
        #print p.contents[0].strip()
        #second = p.findAll('p')
        #for s in second:
        #    all_string = str(s)
        #    all_string
    #for p in position:
    #    #print p.text.split('\n')[0]#.encode('utf-8')
    #    print p.text
    #for d in draft:
    #    draft_l.append(d.contents)
    #    print d.contents

    #df.loc[count] = pd.Series({'PERSON_ID':i[1], 'PLAYERCODE':i[0], 'POSITION':p.contents, 'DRAFT':d.contents})
    

