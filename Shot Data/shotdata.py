import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy.stats as stats

# Load the Data
shooting_data = pd.read_csv("C:\Users\Ohio\Downloads\play-index.csv")
shooting_data2 = pd.read_csv("C:\Users\Ohio\Downloads\play-index_psl_finder.cgi_stats.csv")
shooting_data3 = pd.read_csv("C:\Users\Ohio\Downloads\play-index_psl_finder.cgi_stats-2.csv")

# A Bunch of Data Munging
frames = [shooting_data, shooting_data2, shooting_data3]
final_data = pd.concat(frames)
data = final_data.drop(final_data.columns[[0,3,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31]], axis=1)
data.index = range(0, len(data))
data = data.rename(index=str, columns={"Unnamed: 1" : "Player", "Unnamed: 2" : "Year", "Unnamed: 4" : "Team", "Per 40 Minutes.5" : "Att", "Shooting.2" : "Acc"})
data = data.drop(data.index[0], axis=0)
data = data[data.Player != 'Player']
data[['Att', 'Acc']] = data[['Att', 'Acc']].apply(pd.to_numeric)
data['Acc'] = data['Acc'].fillna(0)

#Create new series's in dataframe
data['sAtt'] = pd.Series(data.Att.apply(lambda x: x / data.Att.max()), name='sAtt', index=data.index)
data['sAcc'] = pd.Series(data['Acc'].divide(data.loc[data.Att > 1, 'Acc'].max()), name='sAcc', index=data.index)
data['F-Score'] = pd.Series(2*data['sAtt']*data['sAcc']/(data['sAtt'] + data['sAcc']), name='F-Score', index=data.index)
data['F-Score'] = data['F-Score'].fillna(0)
data['z'] = pd.Series((data['F-Score'] - data['F-Score'].mean())/data['F-Score'].std(), index=data.index)
data['zShot'] = pd.Series((data.loc[data.Att > 1.0, 'F-Score'] - data.loc[data.Att > 1.0, 'F-Score'].mean())/data.loc[data.Att > 1.0, 'F-Score'].std(), index=data.index)

#import info with players height
height1 = pd.read_csv("C:\Users\Ohio\Downloads\play-index_psl_finder.cgi_stats (1).csv")
height2 = pd.read_csv("C:\Users\Ohio\Downloads\play-index_psl_finder.cgi_stats (3).csv")
height3 = pd.read_csv("C:\Users\Ohio\Downloads\play-index_psl_finder.cgi_stats (4).csv")

#more data munging
second_frame = [height1, height2, height3]
height = pd.concat(second_frame)
height = height.drop(height.columns[[0, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]], axis=1)
height = height[height.iloc[:, 0] != 'Player']
height = height.rename(index=str, columns={"Unnamed: 1" : "Player", "Unnamed: 2" : "Height"})

#merge the data with existing dataframe, re-order for appearance
data = data.merge(height, how='left')
data = data[['Player', 'Year', 'Team', 'Height', 'Att', 'Acc', 'sAtt', 'sAcc', 'F-Score', 'z', 'zShot']]

#convert height to inches so it can be used in calculations
data['Height'] = data.Height.str.split("-").apply(lambda x: int(x[0]) * 12 + int(x[1]))

#create new dataframes for pg's, wings,and frontcourt players based on heights
pg, wing, frontcourt = data[data['Height'] < 76].copy(), data[(data['Height'] >= 76) & (data['Height'] <=80)].copy(), data[data['Height'] > 80].copy()
    
#calculate f-scores based on their position
pg['zPos'] = pd.Series((pg['F-Score'] - pg['F-Score'].mean())/pg['F-Score'].std(), index=pg.index)
wing['zPos'] = pd.Series((wing['F-Score'] - wing['F-Score'].mean())/wing['F-Score'].std(), index=wing.index)
frontcourt['zPos'] = pd.Series((frontcourt['F-Score'] - frontcourt['F-Score'].mean())/frontcourt['F-Score'].std(), index=frontcourt.index)

#merge new 'zPos' scores into original dataframe
new_frames = [pg, wing, frontcourt]
to_merge = pd.concat(new_frames)
data = data.merge(to_merge, how='left')

#begin plot for sAcc vs sAtt
a = data.loc[data.Att > 1, 'sAcc']
b = data.loc[data.Att > 1, 'sAtt']

plt.hold(True)

plt.figure(0)
plt.hist(a, 10, alpha=0.5, label='Accuracy')
plt.hist(b, 10, alpha=0.5, label='Attempts')
plt.legend(loc='upper right')
plt.xlabel('Standardized Score')
plt.ylabel('Frequency')
plt.show()

#begin scatter plot for leaguewide F-Scores
plt.figure(1)
plt.scatter(x=data['sAtt'], y=data['sAcc'], s=data['F-Score']*450, c=np.random.rand(len(data)), alpha=0.5)
plt.title('F-Score Distribution for the Entire League')
plt.xlabel('Standardized Attempts')
plt.ylabel('Standardized Accuracy')
plt.show()



chi = data[data['Team'] == 'CHI']
chi.sort_values(by='F-Score', ascending=False)

def shade_curve(scores):
    plt.figure()
    x = np.linspace(-3.5, 3.5, 100)
    mu, sigma = 0, 1
    plt.plot(x,stats.norm.pdf(x, mu,sigma))
    
    for score in scores:
        
        x_space = np.linspace(score-0.1, score+0.1, 10)
        plt.fill_between(x_space, stats.norm.pdf(x_space))
        


          
          
         
  

        
    

