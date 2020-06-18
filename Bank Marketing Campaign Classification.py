#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:38:58 2020

@author: Abhijit Sudhakar, Dhanashree Chavan, Okan Yazmacilar, Preeti Bhole
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 6)

Term = pd.read_csv("~/Desktop/FinalProject/bank-full.csv", sep=';')

Term.head()
Term.tail()

# Check data types
Term.info()

# Total number of rows and columns
Term.shape

# Rows containing duplicate data
duplicateDFRow = Term[Term.duplicated()]
print(duplicateDFRow)

# Rows containing duplicate data
duplicate_rows_df = Term[Term.duplicated()]
print("duplicate rows: ", duplicate_rows_df.shape)

# Check for values for each column
Term.count()

# Check for missing values
Term.isnull().sum()

# rename default column to credit
Term = Term.rename(columns={'default': 'credit', 'housing': 'housingloan', 'loan': 'personalloan', 'y': 'termdeposit'})

# Summary for numeric features
Term.describe()

# Check target variable distribution
plt.bar(['No', 'Yes'], Term.termdeposit.value_counts().values)
plt.title('Target Variable', fontsize=14)
plt.xlabel('Classes')
plt.ylabel('Amount')
plt.show()

# Target variable is extremely imbalanced. This is important to remember when performing classification and
# evaluation, because even without using of machine learning I can make predictions with roughly 90% accuracy just by
# guessing none of the clients subscribed to the term deposit. Since we are focused only on the clients that said
# 'Yes', chances to get predictions are very thin.

cat_var = ['job', 'marital', 'education', 'credit', 'housingloan', 'month', 'personalloan', 'poutcome', 'termdeposit']

for i in cat_var:
    print(Term[i].value_counts())

# Visualise data

# Workplace
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="job", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Job Frequency distribution wrt term deposit")
sn.set_xticklabels(Term.job.value_counts().index, rotation=30)
sn.legend(loc='upper right')
plt.show()

# marital
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="marital", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Marital Status distribution wrt term deposit")
sn.set_xticklabels(Term.marital.value_counts().index, rotation=30)
sn.legend(loc='upper right')
plt.show()

# education
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="education", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Education Status distribution wrt term deposit")
sn.set_xticklabels(Term.education.value_counts().index, rotation=30)
sn.legend(loc='upper right')
plt.show()

# credit
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="credit", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Credit Status distribution wrt term deposit")
sn.set_xticklabels(Term.credit.value_counts().index, rotation=30)
sn.legend(loc='upper right')
plt.show()

# housingloan
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="housingloan", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Housing Loan Status distribution wrt term deposit")
sn.set_xticklabels(Term.housingloan.value_counts().index, rotation=30)
sn.legend(loc='upper right')
plt.show()

# personalloan
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="personalloan", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Personal Loan Status distribution wrt term deposit")
sn.set_xticklabels(Term.personalloan.value_counts().index, rotation=30)
sn.legend(loc='upper right')
plt.show()

# contact
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="contact", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Contact Status distribution wrt term deposit")
sn.set_xticklabels(Term.contact.value_counts().index, rotation=30)
sn.legend(loc='upper right')
plt.show()

# month
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="month", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Month Status distribution wrt term deposit")
sn.set_xticklabels(Term.month.value_counts().index, rotation=30)
sn.legend(loc='upper right')
plt.show()

# We can see that most calls were made in May month and least were made in December month. That's a good
# information because now we can focus that in which month we need to approach the clients the most.

# Previous contacts vs termdeposit
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="previous", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Number of contacts performed earlier wrt term deposit")
sn.set_xticklabels(Term.previous.value_counts().index)
sn.legend(loc='upper right')
plt.show()

# Poutcome Status
f, ax = plt.subplots(figsize=(12, 8))
sn = sns.countplot(x="poutcome", hue="termdeposit", data=Term, palette="Set2")
sn.set_title("Poutcome status distribution wrt term deposit")
sn.set_xticklabels(Term.poutcome.value_counts().index)
sn.legend(loc='upper right')
plt.show()

# Check Age Distribution
sns.set()
age_plot = sns.distplot(Term['age'], bins=10)
age_plot.set_title('Distribution of age')

# Check Pdays Distribution
sns.set()
age_plot = sns.distplot(Term['pdays'], bins=10)
age_plot.set_title('Distribution of pdays')

# Most of the customers are of 30-50 years of age group.

# Encoding target variable
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
Term['termdeposit'] = encoder.fit_transform(Term['termdeposit'])
Term.head()

# Finding the relations between the variables.
sns.set()
plt.figure(figsize=(20, 10))
c = Term.corr()
sns.heatmap(c, cmap="Accent", annot=True)

# Since we do not need the information of when the customer was contacted, we will drop these features from the
# analysis. Also, poutcome maximum impact was for 'unknown'. So we will remove that from our dataset as well.
Term.drop(['age', 'day', 'previous', 'campaign', 'contact', 'poutcome', ], axis=1, inplace=True)
Term.info()

# since duration is only co-related with o/p variable, we will remove others from further analysis.
# https://towardsdatascience.com/exploratory-data-analysis-in-python-c9a77dfa39ce

X = Term[
    ['balance', 'pdays', 'job', 'marital', 'education', 'credit', 'housingloan', 'personalloan',
     'month', 'duration']]
Y = Term[['termdeposit']]

# Feature Engineering
# We need to encode the categorical variables to train the model
# We can see that all the variables are ordinal categorical data type hence we will use get_dummies on pandas dataframe.

encoder_1 = LabelEncoder()

ordinalList = ['education', 'credit', 'housingloan', 'personalloan']

for i in ordinalList:
    X[i] = encoder_1.fit_transform(X[i])

# Dummy encoding using get_dummies

X = pd.get_dummies(X, columns=['job', 'marital', 'month'], drop_first=True)

cat_variables = X.drop(['balance', 'pdays', 'duration'], axis=1)

# Select from categorical variables
from sklearn.feature_selection import SelectKBest, chi2  # for chi-squared feature selection

sf = SelectKBest(chi2, k='all')
sf_fit = sf.fit(cat_variables, Y)
# print feature scores
for i in range(len(sf_fit.scores_)):
    print(' %s: %f' % (cat_variables.columns[i], sf_fit.scores_[i]))

dfscores = pd.DataFrame(sf_fit.scores_)
dfcolumns = pd.DataFrame(cat_variables.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(12, 'Score'))  # print 10 best features

# plot the scores
datset = pd.DataFrame()
datset['feature'] = cat_variables.columns[range(len(sf_fit.scores_))]
datset['scores'] = sf_fit.scores_
datset = datset.sort_values(by='scores', ascending=True)
sns.set()
sns.barplot(datset['scores'], datset['feature'], color='blue')
sns.set_style('whitegrid')
plt.ylabel('Categorical Feature')
plt.xlabel('Score', fontsize=18)
plt.show()

# We will keep only highest scored categorical predictors
X = X[['balance', 'pdays', 'duration', 'month_mar', 'month_oct', 'month_sep', 'housingloan',
       'month_may', 'job_retired', 'job_student', 'month_dec', 'job_blue-collar', 'personalloan']]

# *********************** Random Feature Selection ********************
# We will validate our selection by running random feature selection technique and check if there are any more
# predictor variables that we need to remove

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X, Y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

# Since data is imbalanced hence we need to balance it.
from imblearn.over_sampling import SMOTE

# Implementing over-sampling for handling imbalanced data
smote = SMOTE(random_state=0)
X_res, y_res = smote.fit_resample(X, Y)

# Let's see how the data changed after sampling
plt.bar(['0', '1'], y_res.termdeposit.value_counts().values)
plt.title('Target Variable', fontsize=14)
plt.xlabel('Classes')
plt.ylabel('Amount')
plt.show()

# Splitting data into separate training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.4, random_state=0)

# check the shape of X_train and X_test
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# *********************** Decision Tree ********************

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# create a dictionary of all values we want to hypertune
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15), 'min_samples_split': [2, 3, 4]}

# decision tree model
dtree_model = DecisionTreeClassifier(random_state=200)

# use gridsearch to test all values
grid_search_cv = GridSearchCV(dtree_model, param_grid, cv=10)
grid_search_cv.fit(X_train, y_train)

# Once we have fit the grid search cv model with training data, we will simply ask what worked best for us
# Get the best parameters
grid_search_cv.best_params_

# Creating decision tree model with best parameters
dt = DecisionTreeClassifier(criterion='gini', max_depth=12, min_samples_split=3)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

# Check accuracy score
import sklearn.metrics as metrics

acc_dt = metrics.accuracy_score(y_pred, y_test)
print('The accuracy of the Decision Tree is', acc_dt)

print(cm)

from sklearn.metrics import f1_score
print("F1 score:", f1_score(y_pred, y_test))

# ********************** RANDOM FOREST ***************************
# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# create a dictionary of all values we want to hypertune
param_grid_random = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15), 'min_samples_split': [2, 3, 4],
                     'n_estimators': np.arange(10, 20)}

# random forest tree model
rforest_model = RandomForestClassifier()

# use RandomizedSearchCV to test all values
ran_grid_search_cv = RandomizedSearchCV(rforest_model, param_grid_random, cv=10, n_iter=100, random_state=150)
ran_grid_search_cv.fit(X_train, np.ravel(y_train))

# Once we have fit the RandomizedSearchCV model with training data, we will simply ask what worked best for us
# Get the best parameters
ran_grid_search_cv.best_params_

# Creating random foret tree model with best parameters
rf = RandomForestClassifier(criterion='gini', max_depth=14, min_samples_split=3, n_estimators=12)
rf.fit(X_train, np.ravel(y_train))
y_pred = rf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Check accuracy
import sklearn.metrics as metrics

acc_rf = metrics.accuracy_score(y_pred, y_test)
print('The accuracy of the Random Forest Tree is', acc_rf)

# ********************** LOGISTIC REGRESSION ***************************
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()
print(result.summary2())

# Now we will do model fitting
logreg.fit(X_train, np.ravel(y_train))
y_pred = logreg.predict(X_test)
acc_lg = logreg.score(X_test, y_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(acc_lg))

# Confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import f1_score
print("F1 score:", f1_score(y_pred, y_test))

# ********************** COMPARISON ***************************

# We will plot bar chart of all three models for comparison
accuracy = [acc_rf, acc_dt, acc_lg]
names = ('acc_rf', 'acc_dt', 'acc_lg')
sns.set()
y_pos = np.arange(len(accuracy))
plt.bar(y_pos, accuracy, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.ylabel('Accuracy')
plt.title('Accuracy of all three models')
plt.show()
