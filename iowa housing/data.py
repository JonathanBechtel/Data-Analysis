import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from sklearn.cross_validation import train_test_split

train = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_train.csv", index_col='Id')
test = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_test.csv", index_col='Id')

train['SalePrice'] = pp.scale(np.log(train.SalePrice))
y = train.SalePrice
del train['SalePrice']
data = pd.concat([train, test])

data['MSSubClass'] = data.MSSubClass.apply(lambda x: str(x))
numerical_feats = data.dtypes[data.dtypes != 'object'].index
data[numerical_feats] = data[numerical_feats].fillna(data[numerical_feats].median())
data[numerical_feats] = np.log1p(data[numerical_feats])
data[numerical_feats] = pp.scale(data[numerical_feats])
data = pd.get_dummies(data, drop_first=True)