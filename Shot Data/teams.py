import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy.stats as stats
import shotdata as sd

#load and munge data for 3PA/G
teams = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-2016.csv")
teams = teams.drop(teams.columns[[0,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25]], axis=1)
teams = teams.rename(index=str, columns={"Unnamed: 1" : "Season", "Unnamed: 2" : "Team", "Team Per Game.6" : "Att"})
teams = teams[teams.Season != 'Season']
teams['Team'] = teams['Team'].str.replace('*', '')

#load and munge data for 3P%
teams2 = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-2016-3P.csv")
teams2 = teams2.drop(teams2.columns[[0,1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]], axis=1)
teams2 = teams2.rename(index=str, columns={"Unnamed: 2" : "Team", "Team Shooting.2" : "Acc"})
teams2 = teams2[teams2.Team != 'Tm']
teams2['Team'] = teams2['Team'].str.replace('*', '')

#merge and create data for F-Score, Z, etc
teams = teams.merge(teams2, how='left')
teams[['Att', 'Acc']] = teams[['Att', 'Acc']].apply(pd.to_numeric)
teams['sAtt'] = teams['Att'].apply(lambda x: x / teams.Att.max())
teams['sAcc'] = teams['Acc'].apply(lambda x: x / teams.Acc.max())
teams['F-Score'] = pd.Series((2*teams['sAtt'] * teams['sAcc'])/ (teams['sAtt'] + teams['sAcc']), index=teams.index)
teams['z'] = pd.Series((teams['F-Score'] - teams['F-Score'].mean())/ teams['F-Score'].std(), index=teams.index)

#load and munge data for 3PA/G in 2005-2006
teams06 = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-2006.csv")
teams06 = teams06.drop(teams06.columns[[0,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25]], axis=1)
teams06 = teams06.rename(index=str, columns={"Unnamed: 1" : "Season", "Unnamed: 2" : "Team", "Team Per Game.6" : "Att"})
teams06 = teams06[teams06.Season != 'Season']
teams06['Team'] = teams06['Team'].str.replace('*', '')

#load and munge data for 3P% in 2005-2006
teams062 = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-2006-3P.csv")
teams062 = teams062.drop(teams062.columns[[0,1,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]], axis=1)
teams062 = teams062.rename(index=str, columns={"Unnamed: 2" : "Team", "Team Shooting.2" : "Acc"})
teams062 = teams062[teams062.Team != 'Tm']
teams062['Team'] = teams062['Team'].str.replace('*', '')

#merge and convert strings to floats
teams06 = teams06.merge(teams062, how='left')
teams06[['Att', 'Acc']] = teams06[['Att', 'Acc']].apply(pd.to_numeric)


#load and munge data for teams from 2006 - 2016 3PA/G
historical = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-96-16.csv")
historical2 = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-96-16-2.csv")
historical3 = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-96-16-3.csv")
historical4 = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-96-16-4.csv")

#concatenate into one dataframe
frame = [historical, historical2, historical3, historical4]
historical = pd.concat(frame)

#drop unnecessary columns, rename, and remove unnecessary rows
historical = historical.drop(historical.columns[[0,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25]], axis=1)
historical = historical.rename(index=str, columns={"Unnamed: 1" : "Season", "Unnamed: 2" : "Team", "Team Per Game.6" : "Att"})
historical = historical[historical.Season != 'Season']
historical['Team'] = historical['Team'].str.replace('*', '')


#load data for teams from 2006 - 2016 3P%
bug = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-96-16-3P.csv")
bug2 = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-96-16-3P-2.csv")
bug3 = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-96-16-3P-3.csv")
bug4 = pd.read_csv("C:\Users\Ohio\Downloads\play-index-teams-96-16-3P-4.csv")

#concat into one document
frame2 = [bug, bug2, bug3, bug4]
bug = pd.concat(frame2)

#munge to rename, remove unnecessary columns and rows, properly format
bug = bug.drop(bug.columns[[0,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]], axis=1)
bug = bug.rename(index=str, columns={"Unnamed: 1" : "Season","Unnamed: 2" : "Team", "Team Shooting.2" : "Acc"})
bug = bug[bug.Team != 'Tm']
bug['Team'] = bug['Team'].str.replace('*', '')

historical = historical.merge(bug, how='left')
historical[['Att', 'Acc']] = historical[['Att', 'Acc']].apply(pd.to_numeric)
historical['sAtt'] = historical['Att'].apply(lambda x: x / historical.Att.max())
historical['sAcc'] = historical['Acc'].apply(lambda x: x / historical.Acc.max())
historical['F-Score'] = pd.Series((2*historical['sAtt'] * historical['sAcc'])/ (historical['sAtt'] + historical['sAcc']), index=historical.index)
historical['z'] = pd.Series((historical['F-Score'] - historical['F-Score'].mean())/ historical['F-Score'].std(), index=historical.index)

#Create Plots & Graphs
plt.hold(True)

#plot for scatter chart comparing F-Scores of teams in 2015-2016 season
plt.figure(0)
plt.scatter(x=teams['sAtt'], y=teams['sAcc'], s=teams['F-Score']*450, c=np.random.rand(len(teams)), alpha=0.5)
plt.title('F-Score Distribution for Teams 2015-2016')
plt.xlabel('Standardized Attempts')
plt.ylabel('Standardized Accuracy')
plt.show()

#plot for histogram comparing distribution of 3P shots from 1996-2016
plt.figure(1)
plt.hist(teams['Att'], 10, alpha=0.5, label='2015-2016')
plt.hist(teams06['Att'], 10, alpha=0.5, label='2005-2006')
plt.legend(loc='upper right')
plt.xlabel('Three Point Attempts')
plt.ylabel('Frequency')
plt.title('Changes in 3 Point Frequency Over 10 Years')
plt.show()

#plot for histogram comparing distribution of 3P shots from 1996-2016
plt.figure(2)
plt.hist(teams['Acc'], 10, alpha=0.5, label='2015-2016')
plt.hist(teams06['Acc'], 10, alpha=0.5, label='2005-2006')
plt.legend(loc='upper right')
plt.xlabel('Three Point Accuracy')
plt.ylabel('Frequency')
plt.title('Change in 3 Point Accuracy Over 10 Years')
plt.show()

plt.figure(3)
plt.scatter(x=historical['sAtt'], y=historical['sAcc'], s=historical['F-Score']*400, c = np.random.rand(len(historical)), alpha=0.5)
plt.title('F-Score Distribution for Teams 2006-2016')
plt.xlabel('Standardized Attempts')
plt.ylabel('Standardized Accuracy')
plt.show()