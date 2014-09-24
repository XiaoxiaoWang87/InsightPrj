import pandas as pd


df = pd.read_csv('position_draft.csv',sep='\t')

for index, row in df.iterrows():
    #if row["DRAFT"] == '[]':
    if row["POSITION"] == '[]':
        print row["PLAYERCODE"]
