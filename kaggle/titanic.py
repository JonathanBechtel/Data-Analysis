import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

#Read files into the program
test = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\Titanic Survivors\_test.csv", index_col='PassengerId')
train = pd.read_csv("C:\Users\Ohio\Documents\Kaggle\Titanic Survivors\_train.csv", index_col='PassengerId')
y = train['Survived']
del train['Survived']

train = pd.concat([train, test])

#Drop unnecessary columns
train = train.drop(train.columns[[6,9]], axis=1)


#Encode categorical data to numbers for further modification
train['Sex'] = LabelEncoder().fit_transform(train.Sex)
#train['Cabin'] = LabelEncoder().fit_transform(pd.notnull(train.Cabin))
train['Family_Size'] = train.SibSp + train.Parch
train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.replace(np.nan, x.median())) 
train.iloc[1043, 6] = 7.90

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
train = train[['Pclass', 'Sex', 'Greeting', 'SibSp', 'Parch', 'Family_Size', 'Age', 'Fare']]

#Categorical coding for data with more than two labels
Greetings = pd.get_dummies(train['Greeting'], prefix='Greeting', drop_first=True)
Pclass = pd.get_dummies(train['Pclass'], prefix='Passenger Class', drop_first=True)

#Scale Continuous Data
train['SibSp_scaled'] = (train.SibSp - train.SibSp.mean())/train.SibSp.std()
train['Parch_scaled'] = (train.Parch - train.Parch.mean())/train.Parch.std()
train['Family_scaled'] = (train.Family_Size - train.Family_Size.mean())/train.Family_Size.std()
train['Age_scaled'] = (train.Age - train.Age.mean())/train.Age.std()
train['Fare_scaled'] = (train.Fare - train.Fare.mean())/train.Fare.std()


#Drop unmodified data since it's no longer needed
train = train.drop(train.columns[[0,2,3,4,5,6,7]], axis=1)

#Concat modified data to be used for analysis, set to X and y values
train = pd.concat([train, Greetings, Pclass], axis=1)

test = train.iloc[891:]
X = train[:891]

#Create cross - validation set 
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.6)

clf = LogisticRegression(solver='lbfgs', class_weight='balanced')
 
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
    plt.show()
    clf.C = Cs[error.index(max(error))]
    print("Ideal value of C is %d" % (Cs[error.index(max(error))]))
    print(max(error))
 
 
#Sets logistic regression to the value of C that corresponds to the highest score on the CV set  
find_C(X_val, y_val)
clf.fit(X, y)

answer = pd.DataFrame(clf.predict(test), index=test.index, columns=['Survived'])
answer.to_csv('C:\Users\Ohio\Documents\Kaggle\Titanic Survivors\_answer.csv')
coef = pd.DataFrame({'Variable': train.columns, 'Coefficient': clf.coef_[0]})