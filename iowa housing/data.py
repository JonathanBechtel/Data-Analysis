import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
import scipy.stats as stats
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge, LassoCV
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from statsmodels.stats.outliers_influence import OLSInfluence as inf
import statsmodels.api as sm
from sklearn.covariance import EllipticEnvelope

#load data
train = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_train.csv", index_col='Id')
test = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_test.csv", index_col='Id')

#combine train, test for munging, transform and separate answer values for later analysis
y = np.log1p(train.SalePrice)
del train['SalePrice']
data = pd.concat([train, test])

#convert to string because it's really a categorical variable
data['MSSubClass'] = data.MSSubClass.apply(lambda x: str(x))

#Fill in Missing Values

#Ugly way of using the mode to fill empty values for the following columns
data.iloc[:, [60,61,8,22,23,24,33,34,35,36,37,41,46,47,52,54,77]] = data.iloc[:, [60,61,8,22,23,24,33,34,35,36,37,41,46,47,52,54,77]].apply(lambda x: x.fillna(x.mode()))
data['MSZoning'] = data['MSZoning'].fillna('RM')

#Ugly way of filling in categorical data when missing data likely means no data.
data.iloc[:, [29,30,31,32,58,59,62,63,5,56,57,71,72,73]] = data.iloc[:, [29,30,31,32,58,59,62,63,5,56,57,71,72,73]].apply(lambda x: x.fillna('None'))

#creates index of features that have continuous data
numerical_feats = data.dtypes[data.dtypes != 'object'].index

#fill remaining values with median of the columns
data[numerical_feats] = data[numerical_feats].fillna(data[numerical_feats].median())

#log transform data for analysis
data[numerical_feats] = np.log1p(data[numerical_feats])

#create dummy variables for categorical data
data = pd.get_dummies(data, drop_first=True)

#Create cross - validation set 
X = data[:1460]
test = data[1460:]

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.6)

clf = Ridge()

#cross validation function to determine optimal value of C
def find_alpha(X, y):
    alphas = [0, 10, 20, 30, 50, 75, 100] #alpha values to use
    score = []
    for alpha in alphas:
        clf.alpha = alpha
        clf.fit(X_train, y_train)
        score.append(clf.score(X, y))
  
    plt.figure()
    plt.plot(alphas, score, marker='x')
    plt.ylim(0.5, 1.0)
    plt.xlabel('Value of Alpha')
    plt.ylabel('Accuracy on Cross Validation Set')
    plt.title('What\'s the Best Value of C?')
    plt.show()
    clf.alpha = alphas[score.index(max(score))]
    print("Ideal value of alpha with Ridge Regression is %g" % (alphas[score.index(max(score))]))
    print("Peak accuracy is %g" % max(score))
    return score

#set the value of Alpha
find_alpha(X_val, y_val)

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
errors['residual_adj'] = np.abs((errors.residual/y) * 100)

#drop outliers, as determined by normalized residuals
X = X.drop([826, 63, 1325, 463, 524, 1454, 969], axis=0)
y = y.drop([826, 63, 1325, 463, 524, 1454, 969], axis=0)

#initiate Lasso Regression, fit to data
lasso = LassoCV(cv=10)
lasso.fit(X, y)
print ("Peak value of R squared with lasso is %g" % lasso.score(X, y))

#create dataframe for answers, write to csv
answer = pd.DataFrame(np.expm1(lasso.predict(test)), index=test.index, columns=['SalePrice'])
answer.to_csv('C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_answer.csv')

#create dataframe containing coefficients, plot it
coeff = pd.DataFrame({'Features': data.columns, 'Coefficients': lasso.coef_})
coeff = coeff.sort_values(by='Coefficients', ascending=False)