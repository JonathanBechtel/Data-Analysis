import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy.stats as stats
import shotdata as sd

#import data from shotdata, create new df for Chicago Bulls
chi = sd.data[sd.data.Team == 'CHI']
chi.iloc[5] = sd.data.iloc[5].copy()
chi.iloc[1] = sd.data.iloc[185].copy()
chi.iloc[6] = sd.data.iloc[267].copy()
chi.iloc[8] = sd.data.iloc[218].copy()
chi = chi.append(sd.data.iloc[172].copy())
chi['Team'] = 'CHI'

#Create sub dataframe for barplot
chi_plot=chi.loc[:, ['z', 'zPos']]

plt.hold(True)

#create and modify horizontal barplot
plt.figure(7)
chi_plot.plot.barh()
plt.yticks(np.arange(len(chi)), chi['Player'])
plt.title('Bulls Players vs. The League, Their Position')
plt.xlabel('Standard Deviations Above/Below League Average')

#begin code for various bell curve graphs
x = np.linspace(-3.5, 3.5, 100)
mu, sigma = 0, 1

#Mirotic vs. Position
plt.figure(1)
x1 = np.linspace(chi.iloc[0, 11] -.1, chi.iloc[0, 11] + .1, 10)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.fill_between(x1, stats.norm.pdf(x1))
plt.title('Nikola Mirotic vs. His Position')

#Rajon Rondo vs. Position
plt.figure(2)
x2 = np.linspace(chi.iloc[1, 11] -.1, chi.iloc[1, 11] + .1, 10)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.fill_between(x2, stats.norm.pdf(x2))
plt.title('Rajon Rondo vs. His Position')

#Doug McDermott vs. Position
plt.figure(3)
x3 = np.linspace(chi.iloc[2, 11] -.1, chi.iloc[2, 11] + .1, 10)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.fill_between(x3, stats.norm.pdf(x3))
plt.title('Doug McDermott vs. His Position')

#Mirotic vs. Portis - Position
plt.figure(4)
x4 = np.linspace(chi.iloc[7, 11] -.1, chi.iloc[7, 11] + .1, 10)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.fill_between(x4, stats.norm.pdf(x4))
plt.fill_between(x1, stats.norm.pdf(x1))
plt.title('Mirotic and Portis vs. Their Position')

#Mirotic vs. Portis - Shooters
plt.figure(5)
x5 = np.linspace(chi.iloc[7, 10] -.1, chi.iloc[7, 10] + .1, 10)
x6 = np.linspace(chi.iloc[0, 10] - .1, chi.iloc[0, 10] + .1, 10)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.fill_between(x5, stats.norm.pdf(x5))
plt.fill_between(x6, stats.norm.pdf(x6))
plt.title('Mirotic & Portis vs. Jumpshooters')

#Canaan vs. Backcourt - Position
plt.figure(6)
x7 = np.linspace(chi.iloc[8, 11] -.1, chi.iloc[8, 11] + .1, 10)
x8 = np.linspace(chi.iloc[4, 11] - .1, chi.iloc[4, 11] + .1, 10)
x9 = np.linspace(chi.iloc[5, 11] - .1, chi.iloc[5, 11] + .1, 10)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.fill_between(x2, stats.norm.pdf(x2))
plt.fill_between(x7, stats.norm.pdf(x7))
plt.fill_between(x8, stats.norm.pdf(x8))
plt.fill_between(x9, stats.norm.pdf(x9))
plt.title('Isaiah Canaan vs. The Rest of the Backcourt at Their Position')

plt.show()