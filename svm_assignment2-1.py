# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 07:55:54 2023

@author: Jungyu Lee, 301236221

Assignment 2 - SVM 

Exercise 1
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
    data_jungyu = pd.read_csv(file_path)
else:
    print(f"File '{file_path}' does not exist.")

# 2. Carryout some initial investigations:
  # a. Check the names and types of columns.
data_jungyu.columns # name
data_jungyu.dtypes # name & types
  # b. Check the missing values.
data_jungyu.isnull().sum() 
  # c. Check the statistics of the numeric fields (mean, min, max, median, count..etc.)
data_jungyu.describe().T
  # d. In your written response, write a paragraph explaining your findings about each column.

# [Pre-process and visualize the data]
# 3. Replace the '?' mark in the 'bare' column by np.nan and change the type to 'float'
data_jungyu['bare'] = data_jungyu['bare'].replace('?', np.nan).astype(float)

# 4. Fill any missing data with the median of the column.
data_jungyu['bare'].median() # 1.0
data_jungyu['bare'].fillna(data_jungyu['bare'].median(), inplace=True)

# 5. Drop the ID column
data_jungyu.drop('ID', axis=1, inplace=True)

# 6. Using Pandas, Matplotlib, seaborn (you can use any or a mix) generate 3-5 plots 
#    and add them to your written response explaining what are the key insights and findings from the plots.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Countplot
sns.countplot(data=data_jungyu, x='class', ax=axes[0, 0])
axes[0, 0].set_xlabel('Class')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Distribution of Classes')

# Boxplot
sns.boxplot(data=data_jungyu, x='class', y='thickness', ax=axes[0, 1])
axes[0, 1].set_xlabel('Class')
axes[0, 1].set_ylabel('Thickness')
axes[0, 1].set_title('Thickness by Class')

# Violinplot
sns.violinplot(data=data_jungyu, x='class', y='Epith', ax=axes[1, 0])
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_ylabel('Epith')
axes[1, 0].set_title('Epith by Class')

# Scatterplot
sns.scatterplot(data=data_jungyu, x='size', y='shape', hue='class', ax=axes[1, 1])
axes[1, 1].set_xlabel('Size')
axes[1, 1].set_ylabel('Shape')
axes[1, 1].set_title('Size vs. Shape with Class')

plt.tight_layout()
plt.show()

# 7. Separate the features from the class.
X = data_jungyu.drop('class', axis=1)
y = data_jungyu['class']

# 8. Split your data into train 80% train and 20% test, use the last two digits of your student number for the seed. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# [Build Classification Models]
# [Support vector machine classifier with linear kernel]

# 9. Train an SVM classifier using the training data, set the kernel to linear and set the regularization parameter to C= 0.1. Name the classifier clf_linear_firstname.
from sklearn.svm import SVC

clf_linear_jungyu = SVC(kernel='linear', C=0.1, random_state=21)
clf_linear_jungyu.fit(X_train, y_train)

# 10. Print out two accuracy score one for the model on the training set i.e. X_train, y_train and the other on the testing set i.e. X_test, y_test. Record both results in your written response.
from sklearn.metrics import accuracy_score
y_train_pred = clf_linear_jungyu.predict(X_train)
accuracy_score(y_train, y_train_pred)

y_test_pred = clf_linear_jungyu.predict(X_test)
accuracy_score(y_test, y_test_pred)

# 11. Generate the accuracy matrix. Record the results in your written response.
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred)

# [Support vector machine classifier with 'rbf' kernel]
# 12. Repeat steps 9 to 11, in step 9 change the kernel to “rbf” and do not set any value for C. 
clf_rbf_jungyu = SVC(kernel='rbf', random_state=21)
clf_rbf_jungyu.fit(X_train, y_train)

y_train_pred_rbf = clf_rbf_jungyu.predict(X_train)
accuracy_score(y_train, y_train_pred_rbf)

y_test_pred_rbf = clf_rbf_jungyu.predict(X_test)
accuracy_score(y_test, y_test_pred_rbf)

confusion_matrix(y_test, y_test_pred_rbf)

# 13. Repeat steps 9 to 11, in step 9 change the kernel to "poly" and do not set any value for C.
clf_poly_jungyu = SVC(kernel='poly', random_state=21)
clf_poly_jungyu.fit(X_train, y_train)

y_train_pred_poly = clf_poly_jungyu.predict(X_train)
accuracy_score(y_train, y_train_pred_poly)

y_test_pred_poly = clf_poly_jungyu.predict(X_test)
accuracy_score(y_test, y_test_pred_poly)

confusion_matrix(y_test, y_test_pred_poly)

# 14. Repeat steps 9 to 11, in step 9 change the kernel to "sigmoid" and do not set any value for C.
clf_sigmoid_jungyu = SVC(kernel='sigmoid', random_state = 21)
clf_sigmoid_jungyu.fit(X_train, y_train)

y_train_pred_sigmoid = clf_sigmoid_jungyu.predict(X_train)
accuracy_score(y_train, y_train_pred_sigmoid)

y_test_pred_sigmoid = clf_sigmoid_jungyu.predict(X_test)
accuracy_score(y_test, y_test_pred_sigmoid)

confusion_matrix(y_test, y_test_pred_sigmoid)

# By now you have the results of four SVM classifiers with different kernels recorded in your written report. 
# Please examine and write a small paragraph indicating which classifier you would recommend and why.



