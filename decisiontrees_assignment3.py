# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:19:54 2023

@author: 8778t

Jungyu Lee 
301236221

Assignment 3 - Decision Trees
"""

import pandas as pd

# 1. load the data
data_jungyu = pd.read_csv("C:/Users/Public/4th/COMP247/W10/student-por.csv", sep=';')

# 2-a. column names and types
data_jungyu.dtypes
# 2-b. missing values
data_jungyu.isnull().sum()
# 2-c. statistics
data_jungyu.describe()
# 2-d. categorical values
categorical_cols = data_jungyu.select_dtypes(include="object").columns
for col in categorical_cols:
    print(col, data_jungyu[col].unique())

## 2-e
    
# 3. create a new target variable
data_jungyu['pass_jungyu'] = ((data_jungyu['G1'] + data_jungyu['G2'] + data_jungyu['G3']) >= 35).astype(int)

# 4. drop the columns G1, G2, G3 permanently.
data_jungyu.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)

# 5. separate features and target variable
features_jungyu = data_jungyu.drop('pass_jungyu', axis=1)
target_jungyu = data_jungyu['pass_jungyu']

## 6. print out the total number of instances in each class
target_jungyu.value_counts()

# 7. create two lists one to save the names of your numeric fields and one to save the names of your categorical fields.
numeric_features_jungyu = data_jungyu.select_dtypes(include=["int64"]).columns.tolist()
cat_features_jungyu = data_jungyu.select_dtypes(include=["object"]).columns.tolist()

# 8. Prepare a column transformer to handle all the categorical variables and convert them into numeric values using one-hot encoding. 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer_jungyu = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(), cat_features_jungyu)],
    remainder="passthrough" # the columns not specified in transformers (those that are not included in cat_features_jungyu) will be passed through without any transformations
)

# 9. prepare a classifier decision tree model
from sklearn.tree import DecisionTreeClassifier
clf_jungyu = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# 10. build a pipeline
from sklearn.pipeline import Pipeline
pipeline_jungyu = Pipeline([
    ("transformer", transformer_jungyu),
    ("classifier", clf_jungyu)
])

# 11. split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train_jungyu, X_test_jungyu, y_train_jungyu, y_test_jungyu = train_test_split(
    features_jungyu, target_jungyu, test_size=0.2, random_state=21)

# 12. fit the training data to the pipeline
pipeline_jungyu.fit(X_train_jungyu, y_train_jungyu)

# 13. cross validate the output on the training data using 10-fold cross validation
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=21)
scores = cross_val_score(pipeline_jungyu, X_train_jungyu, y_train_jungyu, cv=kfold)
print("cross-validation scores:", scores)

## 14. print the mean of the cross-validation scores
print("mean cross-validation score:", scores.mean())

# 15. visualize the tree using Graphviz
import graphviz
from sklearn.tree import export_graphviz
X_transformed = transformer_jungyu.fit_transform(features_jungyu)
encoded_feature_names = transformer_jungyu.named_transformers_["onehot"].get_feature_names_out(cat_features_jungyu)
feature_names = numeric_features_jungyu + list(encoded_feature_names)

dot_data = export_graphviz(
    clf_jungyu,
    out_file=None,
    feature_names=feature_names,
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("pima")
graph

## 16
## 17

## 18. compute accuracy scores for training and testing sets
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

train_score = accuracy_score(y_train_jungyu, pipeline_jungyu.predict(X_train_jungyu))
test_score = accuracy_score(y_test_jungyu, pipeline_jungyu.predict(X_test_jungyu))

print("accuracy score on training set:", train_score)
print("accuracy score on testing set:", test_score)

## 19. use the model to predict the test data and 
# printout the accuracy, precision and recall scores 
# and the confusion matrix. 
y_pred_test = pipeline_jungyu.predict(X_test_jungyu)

accuracy = accuracy_score(y_test_jungyu, y_pred_test)
precision = precision_score(y_test_jungyu, y_pred_test)
recall = recall_score(y_test_jungyu, y_pred_test)
confusion = confusion_matrix(y_test_jungyu, y_pred_test)

print("accuracy score:", accuracy)
print("precision score:", precision)
print("recall score:", recall)
print("confusion matrix:\n", confusion)

# 20. use Randomized grid search fine tune your model
from sklearn.model_selection import RandomizedSearchCV

parameters = {
    'classifier__min_samples_split': range(10, 300, 20),
    'classifier__max_depth': range(1, 30, 2),
    'classifier__min_samples_leaf': range(1, 15, 3)
}

randomized_search = RandomizedSearchCV(
    estimator=pipeline_jungyu,
    param_distributions=parameters,
    scoring='accuracy',
    cv=5,
    n_iter=7,
    refit=True,
    verbose=3,
    random_state=21
)

# 21. fit your training data to the gird search object
randomized_search.fit(X_train_jungyu, y_train_jungyu)
## 22. print out the best parameters 
print("best parameters:", randomized_search.best_params_)
## 23. print out the score of the model 
print("best score:", randomized_search.best_score_)
## 24. print out the best estimator 
print("best estimator:", randomized_search.best_estimator_)

# 25. fit the test data using the fine-tuned model identified during grid search
best_estimator = randomized_search.best_estimator_
y_pred_test = best_estimator.predict(X_test_jungyu)
## 26 print out the precision, recall and accuracy. 
accuracy = accuracy_score(y_test_jungyu, y_pred_test)
precision = precision_score(y_test_jungyu, y_pred_test)
recall = recall_score(y_test_jungyu, y_pred_test)
confusion = confusion_matrix(y_test_jungyu, y_pred_test)

print("accuracy score:", accuracy)
print("precision score:", precision)
print("recall score:", recall)
print("confusion matrix:\n", confusion)

from joblib import dump

# 27. save the model using joblib
dump(best_estimator, 'C:/Users/Public/4th/COMP247/W10/model.pkl')

# 28. save the full pipeline using joblib
dump(pipeline_jungyu, 'C:/Users/Public/4th/COMP247/W10/pipeline.pkl')
