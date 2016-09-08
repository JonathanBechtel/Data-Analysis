import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

test = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\Titanic Survivors\_test.csv")
train = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\Titanic Survivors\_train.csv")
#train = pd.concat([train, test])

train = train.drop(train.columns[[0, 8]], axis=1)
train['Sex'] = LabelEncoder().fit_transform(train.Sex)
train['Cabin'] = LabelEncoder().fit_transform(pd.notnull(train.Cabin))
train.Embarked = train.Embarked.fillna(value='S')
train['Embarked'] = LabelEncoder().fit_transform(train.Embarked)
train['Family_Size'] = train.SibSp + train.Parch
train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.replace(np.nan, x.median())) 

def find_greeting(value):
    greetings = ['Mrs.', 'Miss.', 'Mr.', 'Master.', 'Rev.', 'Don.', 'Dr.', 'Mme.', 'Ms.', 'Major.', 'Lady.', 'Sir.', 'Mlle.', 'Col.', 'Capt.', 'Countess.']
    words = value.split()
    
    for greeting in greetings:
        
        if greeting in words:    
            return greeting
            
train['Greeting'] = train.Name.apply(find_greeting)
train['Greeting'] = train.Greeting.fillna('Mr.')
train['Greeting'] = LabelEncoder().fit_transform(train.Greeting)
del train['Name']

train = train[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Greeting', 'SibSp', 'Parch', 'Family_Size', 'Age', 'Fare', 'Survived']]

#Categorical coding for data with more than two labels
Greetings = pd.get_dummies(train['Greeting'], prefix='Greeting', drop_first=True)
Pclass = pd.get_dummies(train['Pclass'], prefix='Passenger Class', drop_first=True)
Embarked = pd.get_dummies(train['Embarked'], prefix='Port', drop_first=True)

#Scale Continuous Data
train['SibSp_scaled'] = (train.SibSp - train.SibSp.mean())/train.SibSp.std()
train['Parch_scaled'] = (train.Parch - train.Parch.mean())/train.Parch.std()
train['Family_scaled'] = (train.Family_Size - train.Family_Size.mean())/train.Family_Size.std()
train['Age_scaled'] = (train.Age - train.Age.mean())/train.Age.std()
train['Fare_scaled'] = (train.Fare - train.Fare.mean())/train.Fare.std()

X = train.drop(train.columns[[0,3,4,5,6,7,8,9,10]], axis=1)
X = pd.concat([X, Greetings, Pclass, Embarked], axis=1)
y = train.Survived

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.6, random_state=0)

clf = LogisticRegression()
clf.fit(X_train, y_train)
 
def find_C(X, y):
    Cs = np.logspace(-4, 4, 10)
    error = []  
    for C in Cs:
        clf.C = C
        clf.fit(X_train, y_train)
        error.append(clf.score(X, y))
        
    plt.figure()
    plt.semilogx(Cs, error)
    plt.xlabel('Value of C')
    plt.ylabel('Score on Cross Validation Set')
    plt.title('Optimal Value of C')
    
clf.C = 1000