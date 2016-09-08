import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\Titanic Survivors\_train.csv")
train = train.drop(train.columns[[0, 8]], axis=1)
train['Sex'] = LabelEncoder().fit_transform(train.Sex)
train['Cabin'] = LabelEncoder().fit_transform(pd.notnull(train.Cabin))
train.Embarked = train.Embarked.fillna(value='S')
train['Embarked'] = LabelEncoder().fit_transform(train.Embarked)
train['Family Size'] = train.SibSp + train.Parch
train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.replace(np.nan, x.median())) 

def find_greeting(value):
    greetings = ['Mrs.', 'Miss.', 'Mr.', 'Master.', 'Rev.', 'Don.', 'Dr.', 'Mme.', 'Ms.', 'Major.', 'Lady.', 'Sir.', 'Mlle.', 'Col.', 'Capt.', 'Countess.']
    words = value.split()
    
    for greeting in greetings:
        
        if greeting in words:
            
            value = greeting
            
            return value
            
train['Greeting'] = train.Name.apply(find_greeting)
train['Greeting'] = train.Greeting.fillna('Mr.')
train['Greeting'] = LabelEncoder().fit_transform(train.Greeting)
del train['Name']

train = train[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Greeting', 'SibSp', 'Parch', 'Family Size', 'Age', 'Fare', 'Survived']]
