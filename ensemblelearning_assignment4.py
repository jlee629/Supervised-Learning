# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 22:55:34 2023

@author: 8778t

Jungyu Lee 301236221
COMP247 SEC 001
Ensemble Learning
"""

import pandas as pd
import numpy as np
# 1. Load the data 
# 2. Add the column names i.e. add a header record.

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df_jungyu = pd.read_csv("C:/Users/Public/4th/COMP247/W12/pima-indians-diabetes.csv", header=None, names=columns)

# 3.
# a. names and types of columns
df_jungyu.info()
df_jungyu.dtypes

# b. missing values
df_jungyu.isnull().sum()

# c. statistics of the numeric fields
pd.set_option('display.max_columns', None)
df_jungyu.describe()

# d. the categorical values, if any
categorical_cols = df_jungyu.select_dtypes(include="object").columns
for col in categorical_cols:
    print(col, df_jungyu[col].unique())
    
# f. the total number of instances in each class 
df_jungyu['Outcome'].value_counts()

# 4. prepare a standard scaler transformer to transform all the numeric values. 
from sklearn.preprocessing import StandardScaler

transformer_jungyu = StandardScaler()

# 5. split the features from the class.
X_jungyu = df_jungyu.drop(columns=['Outcome']) 
y_jungyu = df_jungyu['Outcome'] 


# 6. split your data into train 70% train and 30% test, use 42 for the seed.  Name the train/test dataframes 
# combine the features and target variable for oversampling only the training data
# split the dataset into training and test sets with stratified sampling, using 30% for test set and seed 42.
from sklearn.model_selection import train_test_split
X_train_jungyu, X_test_jungyu, y_train_jungyu, y_test_jungyu = train_test_split(
    X_jungyu, y_jungyu, test_size=0.3, random_state=42
)

from imblearn.over_sampling import SMOTE
from collections import Counter

# initialize SMOTE for oversampling the minority class
# (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
print(Counter(y_train_jungyu))

# apply SMOTE to only the training set
X_train_jungyu, y_train_jungyu = smote.fit_resample(X_train_jungyu, y_train_jungyu)

# verify the class distribution after oversampling
print("Class distribution after oversampling:")
print(Counter(y_train_jungyu))
print(Counter(y_test_jungyu))

# 7. apply (fit, transform the transformer prepared in step 4 to the features. 
X_train_jungyu_scaled = transformer_jungyu.fit_transform(X_train_jungyu)
X_test_jungyu_scaled = transformer_jungyu.transform(X_test_jungyu)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 8. define 5 classifiers
logistic_jungyu_X = LogisticRegression(max_iter=1400, random_state=42)
random_forest_jungyu_X = RandomForestClassifier(random_state=42)
svm_jungyu_X = SVC(probability=True, random_state=42)
decision_tree_jungyu_X = DecisionTreeClassifier(criterion="entropy", max_depth=42, random_state=42)
extra_trees_jungyu_X = ExtraTreesClassifier(random_state=42)

## hard voting
# 9. define a voting classifier
voting_classifier_jungyu_X = VotingClassifier(
    estimators=[
        ('logistic', logistic_jungyu_X),
        ('random_forest', random_forest_jungyu_X),
        ('svm', svm_jungyu_X),
        ('decision_tree', decision_tree_jungyu_X),
        ('extra_trees', extra_trees_jungyu_X)
    ],
    voting='hard'
)

# 10. fit the training data to the voting classifier
voting_classifier_jungyu_X.fit(X_train_jungyu_scaled, y_train_jungyu)

# 11. predict the first three instances of test data and print the predicted and actual values
classifiers = [logistic_jungyu_X, random_forest_jungyu_X, svm_jungyu_X, decision_tree_jungyu_X, extra_trees_jungyu_X, voting_classifier_jungyu_X]

from sklearn.metrics import accuracy_score

for clf in classifiers:
    clf.fit(X_train_jungyu_scaled, y_train_jungyu)
    predicted_values = clf.predict(X_test_jungyu_scaled)
    accuracy = accuracy_score(y_test_jungyu, predicted_values)
    print(f"{clf.__class__.__name__} Accuracy: {accuracy * 100:.2f}%")

## 14. soft voting
# 9. define a voting classifier with soft voting
voting_classifier_jungyu_X_soft = VotingClassifier(
    estimators=[
        ('logistic', logistic_jungyu_X),
        ('random_forest', random_forest_jungyu_X),
        ('svm', svm_jungyu_X),
        ('decision_tree', decision_tree_jungyu_X),
        ('extra_trees', extra_trees_jungyu_X)
    ],
    voting='soft'  # Set voting to soft
)

# 10. fit the training data to the voting classifier
voting_classifier_jungyu_X_soft.fit(X_train_jungyu_scaled, y_train_jungyu)

# 11. predict the test data and print the accuracy
classifiers_soft = [logistic_jungyu_X, random_forest_jungyu_X, svm_jungyu_X, decision_tree_jungyu_X, extra_trees_jungyu_X, voting_classifier_jungyu_X_soft]

for clf in classifiers_soft:
    clf.fit(X_train_jungyu_scaled, y_train_jungyu)
    predicted_values = clf.predict(X_test_jungyu_scaled)
    accuracy = accuracy_score(y_test_jungyu, predicted_values)
    print(f"{clf.__class__.__name__} Accuracy (soft voting): {accuracy * 100:.2f}%")
    
from sklearn.pipeline import Pipeline

## 15. create two different pipelines
# pipeline #1 : Extra Trees Classifier
pipeline1_jungyu = Pipeline([
    ('scaler', transformer_jungyu),
    ('extra_trees', extra_trees_jungyu_X)
])

# pipeline #2 : Decision Tree Classifier
pipeline2_jungyu = Pipeline([
    ('scaler', transformer_jungyu),
    ('decision_tree', decision_tree_jungyu_X)
])
    
# 16. fit the original data to both pipelines.

pipeline1_jungyu.fit(X_train_jungyu, y_train_jungyu)
pipeline2_jungyu.fit(X_train_jungyu, y_train_jungyu)

# 17. Carry out a 10 fold cross validation for both pipelines set shuffling to true and random_state to 42
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores_pipeline1 = cross_val_score(pipeline1_jungyu, X_train_jungyu, y_train_jungyu, cv=kfold)

# 18. Printout the mean score evaluation
# calculate the mean score for Pipeline #1
mean_score_pipeline1 = np.mean(scores_pipeline1)
print("Mean Score for Pipeline #1:", mean_score_pipeline1)

# calculate the mean score for Pipeline #2 using cross_val_score
scores_pipeline2 = cross_val_score(pipeline2_jungyu, X_train_jungyu, y_train_jungyu, cv=kfold)
mean_score_pipeline2 = np.mean(scores_pipeline2)
print("Mean Score for Pipeline #2:", mean_score_pipeline2)

# 19. predict the test using both pipelines and printout the confusion matrix, precision, recall and accuracy score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

pipelines = [pipeline1_jungyu, pipeline2_jungyu]
pipeline_names = ['Pipeline #1', 'Pipeline #2']

for pipeline, name in zip(pipelines, pipeline_names):
    pipeline.fit(X_train_jungyu, y_train_jungyu)
    y_pred = pipeline.predict(X_test_jungyu)
    
    print(f"\n{name} Evaluation:")
    print("Confusion Matrix:\n", confusion_matrix(y_test_jungyu, y_pred))
    print("Precision:", precision_score(y_test_jungyu, y_pred))
    print("Recall:", recall_score(y_test_jungyu, y_pred))
    print("Accuracy:", accuracy_score(y_test_jungyu, y_pred))

# 21. carry out a randomized grid search on Pipeline #1
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'extra_trees__n_estimators': np.arange(10, 3001, 20),
    'extra_trees__max_depth': np.arange(1, 1001, 2)
}

random_search_jungyu = RandomizedSearchCV(
    pipeline1_jungyu,
    param_distributions=param_dist,
    n_iter=10, # default
    cv=5, # default
    random_state=42,
    n_jobs=-1
)

# 22. fit your training data to the randomized gird search object.
random_search_jungyu.fit(X_train_jungyu, y_train_jungyu)

# 23. print out the best parameters and accuracy score for randomized grid search 
print("Best Parameters:", random_search_jungyu.best_params_)
print("Best Accuracy Score:", random_search_jungyu.best_score_)

# 24
# get the best estimator from the randomized grid search
best_estimator = random_search_jungyu.best_estimator_

# predict the test data using the best estimator
y_pred_best = best_estimator.predict(X_test_jungyu)

# 25. printout the precision, recall and accuracy. 
from sklearn.metrics import precision_score, recall_score, accuracy_score

precision = precision_score(y_test_jungyu, y_pred_best)
recall = recall_score(y_test_jungyu, y_pred_best)
accuracy = accuracy_score(y_test_jungyu, y_pred_best)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
