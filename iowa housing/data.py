import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import OLSInfluence as inf
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import RFECV

#########################################################
#Data Munging and Feature Generation
#########################################################

#load data
train = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_train.csv", index_col='Id')
test = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_test.csv", index_col='Id')

#combine train, test for munging, transform and separate answer values for later analysis
y = np.log1p(train.SalePrice)
del train['SalePrice']
data = pd.concat([train, test])

#convert to string because it's really a categorical variable
data['MSSubClass'] = data.MSSubClass.apply(lambda x: str(x))

#Ugly way of using the mode to fill empty values for the following columns
data.iloc[:, [60,61,8,22,23,24,33,34,35,36,37,41,46,47,52,54,77]] = data.iloc[:, [60,61,8,22,23,24,33,34,35,36,37,41,46,47,52,54,77]].apply(lambda x: x.fillna(x.mode()))

#Most Common value for this column that's paired with MSSubClass
data['MSZoning'] = data['MSZoning'].fillna('RM')

#Ugly way of filling in categorical data when missing data likely means no data.
data.iloc[:, [29,30,31,32,59,62,63,5,56,57,71,72,73]] = data.iloc[:, [29,30,31,32,59,62,63,5,56,57,71,72,73]].apply(lambda x: x.fillna('None'))

#Change categorical data to ordinal data for new interaction terms
#data[['BsmtCond']] = data[['BsmtCond']].replace(to_replace=['TA', 'Gd', 'Fa', 'None', 'Po'], value=[3,4,2,0,1])
#data[['BsmtQual']] = data[['BsmtQual']].replace(to_replace=['TA', 'Gd', 'Fa', 'None', 'Po', 'Ex'], value=[3,4,2,0,1,5])
#data[['FireplaceQu']] = data[['FireplaceQu']].replace(to_replace=['TA', 'Fa', 'Gd', 'Po', 'Ex', 'None'], value=[3,2,4,1,5,0])

#Function to fill in missing values for GarageYrBlt with date house was built.
def YrBlt(series):
    for index, value in enumerate(series):
        if value != value:
            series.iloc[index] = data['YearBuilt'].iloc[index]
         
#Function to categorize YearRemodAdd to see if renovations added any value   
def isRemodeled(value):
        if value <= 0:
            return 'Never'
        elif 0 < value < 6:
            return 'Recently'
        elif 6 <= value < 16:
            return 'Not Recently'
        else:
            return 'Distant Past'

#Function creates a timeseries that numbers each month sequentially
def timeSeries(series1, series2):
        for index, year in enumerate(series2):
            if year == 2006:
                series1.iloc[index] = 0 + data['MoSold'].iloc[index]
            elif year == 2007:
                series1.iloc[index] = 12 + data['MoSold'].iloc[index]
            elif year == 2008:
                series1.iloc[index] = 24 + data['MoSold'].iloc[index]
            elif year == 2009:
                series1.iloc[index] = 36 + data['MoSold'].iloc[index]
            else:
                series1.iloc[index] = 48 + data['MoSold'].iloc[index]
            
#Call the function to fill the values
YrBlt(data.GarageYrBlt)

#creation of new features that capture potential non-linear interactions between important variables
data['interactions'] = data.GrLivArea * data.OverallCond * data.OverallQual
data['interactions2'] = data.Neighborhood + data.SaleCondition
data['garage_added'] = data.GarageYrBlt > data.YearBuilt
data['garage_added'] = data.garage_added.apply(lambda x: 'yes' if x==True else 'no')
data['Remodeled'] = data['YearRemodAdd'] - data['YearBuilt'] 
data['Remodeled'] = data['Remodeled'].apply(isRemodeled)
data['Time_Since_Sale'] = 2016 - data['YrSold']

#zero fill new series and then pass it into timeSeries function to get new values
data['Time_Series'] = pd.Series(np.zeros(len(data)), index=data.index)
timeSeries(data.Time_Series, data.YrSold)

#creates list of features that have continuous data
numerical_feats = data.dtypes[data.dtypes != 'object'].index

#fill remaining values with median of the columns
data[numerical_feats] = data[numerical_feats].fillna(data[numerical_feats].median())

#log transform data for analysis
data[numerical_feats] = np.log1p(data[numerical_feats])

#create dummy variables for categorical data
data = pd.get_dummies(data, drop_first=True)

#######################################################
#Cross Validation, Model Fitting, Prediction
#######################################################

#Create cross - validation set 
X = data[:1460]
test = data[1460:]

#initiate OLS w/ statsmodels to determine outliers
X = sm.add_constant(X)
model = sm.OLS(y, X, missing='raise')
results = model.fit()
outliers = inf(results)

#collect appropriate error terms for each sample in data, create df
residuals = results.resid
cooks_d = outliers.cooks_distance[0]
influence = outliers.influence
errors = pd.DataFrame({'residual': residuals, 'Cooks Distance': cooks_d, 'Influence': influence})

#drop outliers, as determined by Cook's Distance
X = X.drop([826, 524, 1171, 1424, 741, 706, 874, 589, 1001, 411], axis=0)
y = y.drop([826, 524, 1171, 1424, 741, 706, 874, 589, 1001, 411], axis=0)

#drop outliers from errors df to make later analysis easier
errors = errors.drop([826, 524, 1171, 1424, 741, 706, 874, 589, 1001, 411], axis=0)

#initiate Lasso Regression
lasso = LassoCV(cv=10)
lasso.fit(X, y)
print ("Best RMSE with lasso is %g" % np.sqrt(np.mean((lasso.predict(X)-y)**2)))

#create dataframe for answers, write to csv
answer = pd.DataFrame(np.expm1(lasso.predict(test)), index=test.index, columns=['SalePrice'])
answer.to_csv('C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_answer.csv')

#create dataframe containing coefficients, plot it
coeff = pd.DataFrame({'Features': data.columns, 'Coefficients': lasso.coef_})
coeff = coeff.sort_values(by='Coefficients', ascending=False)

########################################################
#  Creation of different plots to visually analyze data
########################################################

#initiate figure
plt.figure(0)

#initiate scatter plot, label it
plt.scatter(lasso.predict(X), y, s=errors.residual*350)
plt.title('Scatter Plot of Housing Prices')
plt.xlabel('Predicted Prices')
plt.ylabel('Actual Prices')   
plt.show()

#histogram of error terms
plt.figure(1)
errors.residual.hist(bins=20)
plt.title('Distribution of errors for predictions')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

##########################################################################
# Feature Graveyard .... Features that didn't add any value to final score
##########################################################################

#data['zoning_interaction'] = data.MSSubClass + data.MSZoning
#data['Fireplace_adj'] = data.Fireplaces * data.FireplaceQu
#data['Total_baths'] = data.HalfBath + data.FullBath
#data['basement_interactions'] = data.TotalBsmtSF * data.BsmtCond * data.BsmtQual