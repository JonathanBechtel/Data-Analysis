import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

#Read files into the program
test = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\Titanic Survivors\_test.csv")
train = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\Titanic Survivors\_train.csv")
#train = pd.concat([train, test])

#Drop unnecessary columns
train = train.drop(train.columns[[0, 8]], axis=1)

#Encode categorical data to numbers for further modification
train['Sex'] = LabelEncoder().fit_transform(train.Sex)
train['Cabin'] = LabelEncoder().fit_transform(pd.notnull(train.Cabin))
train.Embarked = train.Embarked.fillna(value='S')
train['Embarked'] = LabelEncoder().fit_transform(train.Embarked)
train['Family_Size'] = train.SibSp + train.Parch
train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.replace(np.nan, x.median())) 

#Used to create new pd Series from Name data that extracts the greeting used for their name to be used as a separate variable
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

#Reorder columns to group continuous, categorical data together
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

#Drop unmodified data since it's no longer needed
X = train.drop(train.columns[[0,3,4,5,6,7,8,9,10]], axis=1)

#Concat modified data to be used for analysis, set to X and y values
X = pd.concat([X, Greetings, Pclass, Embarked], axis=1)
y = train.Survived

#Create cross - validation set 
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.6, random_state=0)

clf = LogisticRegression(solver='lbfgs')
 
#Function used to determine optimal lambda value for regularization
def find_C(X, y):
    Cs = np.logspace(-4, 4, 10)
    error = []  
    for C in Cs:
        clf.C = C
        clf.fit(X_train, y_train)
        error.append(clf.score(X, y))
  
    plt.figure()
    plt.semilogx(Cs, error, marker='x')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy on Cross Validation Set')
    plt.title('What\'s the Best Value of C?')
    clf.C = Cs[error.index(max(error))]
    print("Ideal value of C is %d" % (Cs[error.index(max(error))]))
 
#Sets logistic regression to the value of C that corresponds to the highest score on the CV set  
find_C(X_val, y_val)