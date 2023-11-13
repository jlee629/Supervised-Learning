# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:42:08 2023


@author: Jungyu Lee, 301236221

Assignment 2 - SVM 

Exercise 2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# [Load & check the data]
# 1. Load the data into a pandas dataframe named data_firstname where first name is you name.
file_path = "C:/Users/Public/Assignment2/breast_cancer.csv"
if os.path.exists(file_path):
    data_jungyu_df2 = pd.read_csv(file_path)
else:
    print(f"File '{file_path}' does not exist.")

# 2. Replace the ‘?’ mark in the ‘bare’ column by np.nan and change the type to ‘float’
data_jungyu_df2['bare'] = data_jungyu_df2['bare'].replace('?', np.nan).astype(float)

# 3. Drop the ID column
data_jungyu_df2.drop('ID', axis=1, inplace=True)

# 4. Separate the features from the class.
X = data_jungyu_df2.drop('class', axis=1)
y = data_jungyu_df2['class']

# 5. Split your data into train 80% train and 20% test use the last two digits of your student number for the seed. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# 6. Using the preprocessing library to define two transformer objects to transform your training data:
 # a. Fill the missing values with the median (hint: checkout SimpleImputer)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
 # b. Scale the data  (hint: checkout StandardScaler)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 7. Combine the two transformers into a pipeline name it num_pipe_firstname.
from sklearn.pipeline import Pipeline
num_pipe_jungyu = Pipeline([
    ('imputer', imputer),  
    ('scaler', scaler)
    ])
# 8. Create a new Pipeline that has two steps the first is the num_pipe_firstname 
#    and the second is an SVM classifier with random state = last two digits of your student number. 
#    Name the pipeline pipe_svm_firstname. (make note of the labels)
from sklearn.svm import SVC
pipe_svm_jungyu = Pipeline([
    ('preprocessing', num_pipe_jungyu),  
    ('svm', SVC(random_state=21))       
    ])

# 9. Take a screenshot showing your num_pipe_firstname object and add it to your written report.

# 10. Define the grid search parameters in an object and name it param_grid, as follows:
 # a. 'svc__kernel': ['linear', 'rbf','poly'],
 # b. 'svc__C':  [0.01,0.1, 1, 10, 100],
 # c. 'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
 # d. 'svc__degree':[2,3]},
# Make sure you replace svc with the label you used in the pipe_svm_firstname for the model
param_grid = {
    'svm__kernel': ['linear','rbf','poly'],
    'svm__C': [0.01, 0.1, 1, 10, 100],
    'svm__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    'svm__degree': [2, 3]
}

# 11. Take a screenshot showing your grid search parameter object and add it to your written report.

# 12. Create a grid search object name it grid_search_firstname with the following parameters:
 # a. estimator= pipe_svm_firstname 
 # b. param_grid=param_grid_svm
 # c. scoring='accuracy' 
 # d. refit = True
 # e. verbose = 3

from sklearn.model_selection import GridSearchCV

grid_search_jungyu = GridSearchCV(
    estimator=pipe_svm_jungyu,
    param_grid=param_grid,
    scoring='accuracy',
    refit=True,
    verbose=3
    )

# 13. Take a screenshot showing your grid search object and add it to your written report.

# 14. Fit your training data to the gird search object. (This will take some time but you will see the results on the console)
grid_search_jungyu.fit(X_train, y_train)

# 15. Print out the best parameters and note it in your written response
grid_search_jungyu.best_params_

# 16. Printout the best estimator and note it in your written response
grid_search_jungyu.best_estimator_

# 17. Fit the test data the grid search object and note it in your written response
grid_search_jungyu.fit(X_test, y_test)
grid_search_jungyu.best_params_
grid_search_jungyu.best_estimator_

# 18. Printout the accuracy score and note it in your written response.
from sklearn.metrics import accuracy_score
y_pred = grid_search_jungyu.predict(X_test)
accuracy_score(y_test, y_pred)

# 19. Create an object that holds the best model i.e. best estimator to an object named best_model_firstname
best_model_jungyu = grid_search_jungyu.best_estimator_

# 20. Save the model using the joblib (dump).
from joblib import dump
dump(best_model_jungyu, 'C:/Users/Public/Assignment2/best_model_jungyu.pkl')
# 21. Save the full pipeline using the joblib – (dump).
dump(pipe_svm_jungyu, 'C:/Users/Public/Assignment2/full_pipeline_jungyu.pkl')
