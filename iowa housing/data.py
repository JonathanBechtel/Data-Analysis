import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
import matplotlib.pyplot as plt
plt.style.use('ggplot')

train = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_train.csv", index_col='Id')
test = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_test.csv", index_col='Id')

y = np.log1p(train.SalePrice)
del train['SalePrice']
data = pd.concat([train, test])

data['MSSubClass'] = data.MSSubClass.apply(lambda x: str(x))

#Fill in Missing Values

#Ugly way of using the mode to fill empty values for the following columns
data.iloc[:, [60,61,8,22,23,24,33,34,35,36,37,41,46,47,52,54,77]] = data.iloc[:, [60,61,8,22,23,24,33,34,35,36,37,41,46,47,52,54,77]].apply(lambda x: x.fillna(x.mode()))
data['LotFrontage'] = data['LotFrontage'].fillna(data.LotFrontage.median())
data['MSZoning'] = data['MSZoning'].fillna('RM')
data.iloc[:, [29,30,31,32,58,59,62,63,5,56,57,71,72,73]] = data.iloc[:, [29,30,31,32,58,59,62,63,5,56,57,71,72,73]].apply(lambda x: x.fillna('None'))

numerical_feats = data.dtypes[data.dtypes != 'object'].index

data[numerical_feats] = data[numerical_feats].fillna(data[numerical_feats].median())
data[numerical_feats] = np.log1p(data[numerical_feats])
data = pd.get_dummies(data, drop_first=True)

#Create cross - validation set 
X = data[:1460]
test = data[1460:]

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.6)

clf = Ridge()

def find_alpha(X, y):
    alphas = [0, 10, 20, 30, 50, 75, 100]
    score = []  
    for alpha in alphas:
        clf.alpha = alpha
        clf.fit(X_train, y_train)
        score.append(clf.score(X, y))
  
    plt.figure()
    plt.plot(alphas, score, marker='x')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy on Cross Validation Set')
    plt.title('What\'s the Best Value of C?')
    plt.show()
    clf.alpha = alphas[score.index(max(score))]
    print("Ideal value of alpha is %g" % (alphas[score.index(max(score))]))
    print(max(score))
    return score


find_alpha(X_val, y_val)
answer = pd.DataFrame(np.expm1(clf.predict(test)), index=test.index, columns=['SalePrice'])
answer.to_csv('C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_answer.csv')