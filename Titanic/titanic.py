#-*- coding: utf-8 -*-

"""
	2015.10.30 김우정
	Titanic: machine learning from disaster
	https://www.kaggle.com/c/titanic
	https://www.dataquest.io/mission/74/getting-started-with-kaggle/
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pd.read_csv("train.csv")

# The titanic variable is available here.
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Find all the unique values for "Embarked".
print(titanic["Embarked"].unique())
titanic["Embarked"] = titanic["Embarked"].fillna(0)
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)


titanic_test = pd.read_csv("test.csv")

# The titanic variable is available here.
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

# Replace all the occurences of male with the number 0.
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# Find all the unique values for "Embarked".
titanic_test["Embarked"] = titanic_test["Embarked"].fillna(0)
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic["Fare"].median())

# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
#Any files you save will be available in the output tab below
submission.to_csv("kaggle.csv", index=False)
                