import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.style.use('ggplot')

app_events = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\TalkingData\_app_events.csv", usecols = [0, 1, 3], dtype = {'is_active': bool} )
app_labels = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\TalkingData\_app_labels.csv")
events = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\TalkingData\events.csv", usecols = [0, 1])
phones = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\TalkingData\phone_brand_device_model.csv")
categories = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\TalkingData\label_categories.csv")
train = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\TalkingData\gender_age_train.csv")
test = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\TalkingData\gender_age_test.csv")

#Drop phone duplicates
phones = phones.drop_duplicates('device_id')
phones['brand'] = LabelEncoder().fit_transform(phones.phone_brand)
phones['device'] = LabelEncoder().fit_transform(phones.device_model)
phones['phone_int'] = phones['device'] * phones['brand'] #interaction effect
phones = phones.drop(phones.columns[[1, 2]], axis=1)

train = train.merge(phones, how='left', on='device_id')
train = train.drop(train.columns[[1, 2]], axis=1)
labels = LabelEncoder().fit(train.group)
train['group'] = labels.transform(train.group)
train = train[['device_id', 'brand', 'device', 'phone_int', 'group']]

test = test.merge(phones, how='left', on='device_id')

X_train, X_val, y_train, y_val = train_test_split(train.iloc[:, [0,1,2,3]], train.group, train_size=0.6, random_state=0)

clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)

labels = LabelEncoder().fit(train.group)
pred = np.zeros((len(y_val), len(labels.classes_)))

Cs = np.logspace(-4, 4, 10)
res = []

#def find_C():
for C in Cs:
    clf.C = C
    pred = clf.predict_proba(X_val)
    res.append(log_loss(y_val, pred))

Xtrain = train.iloc[:, [0,1,2,3]]
y = train.group

clf.fit(Xtrain, y)
Xtest = test.iloc[:, [0,1,2,3]]
final_answer = pd.DataFrame(clf.predict_proba(Xtest), index=test['device_id'], columns=labels.classes_)
final_answer.to_csv('logreg_subm.csv',index=True)
