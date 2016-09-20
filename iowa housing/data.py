import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
import scipy.stats as stats

train = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_train.csv")
test = pd.read_csv("C:\Users\Ohio\Documents\GitHub\data-analysis\iowa housing\_test.csv")

train['SalePrice'] = pp.scale(np.log(train.SalePrice))
y = train.SalePrice
del train['SalePrice']
data = pd.concat([train, test])
