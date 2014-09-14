InsightPrj
==========

Repository for Insight Data Science project. Develope forcasting models using the game log data from NBA.com JSON feed.

	1. Get all player names from "http://stats.nba.com/stats/commonallplayers?LeagueID=00&Season=2013-14&IsOnlyCurrentSeason=0". Create names.json

	2. Run get_names.py twice to get all game logs of all players since 1980, for regular seasons and for playoffs. Create player_post1980_inclusive_log.csv

	3. Run get_allstar.py to get the all-star information (names, etc.). Create allstar_log.csv

	4. Run analyze_gamelog.py to preprocess data and build tables to be ready for SQL database. Create allstar_post1980_sql_log.csv and nonstar_post1980_sql_log.csv

	5. Build mySQL database gamelogdb

	6. Run sql_gamelog.py to write all-star and non-star data into the database (two tables)

	7. Run analysis models here


Q: 
	- I forgot what select_allstar.py does... Probably just some testing example during my development.
	- Also, when I tried step 1 again, I found a slightly different json file. A bit weird but I'm too tired to figure out the exact reason..
