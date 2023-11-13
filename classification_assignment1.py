# -*- coding: utf-8 -*-
"""
Created on Sun May 28 18:40:28 2023

@author: Jungyu Lee, 301236221

Lab Assignment 1: Classfication
"""
# 0. Import basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load & check the data:
# 1. Load the MINST data into a pandas dataframe named MINST_firstname where first name is you name.
from sklearn.datasets import fetch_openml
MINST_jungyu = fetch_openml('mnist_784', version=1)

# 2. List the keys
MINST_jungyu.keys()

# 3. Assign the data to a ndarray named X_firstname where firstname is your first name
X = MINST_jungyu["data"]
X_jungyu = X.to_numpy()

# 4. Assign the target to a variable named y_firstname where firstname is your first name.
y = MINST_jungyu["target"]
y_jungyu = y.to_numpy()

# 5. Print the types of X_firstname and y_firstname.
type(X_jungyu)
X_jungyu.dtype
type(y_jungyu)
y_jungyu.dtype

# 6. Print the shape of X_firstname and y_firstname.
X_jungyu.shape
y_jungyu.shape

# 7. (J) Name the variable some_digit1, some_digit2, some_digit3. Store in these variables the values from X_firstname indexed 7,5,0 in order.
some_digit1 = X_jungyu[7]
some_digit2 = X_jungyu[5]
some_digit3 = X_jungyu[0]

# 8. Use imshow method to plot the values of the three variables you defined in the above point. Note the values in your Analysis report (written response).
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].imshow(some_digit1.reshape(28, 28), cmap='binary')
axes[0].grid(True)
axes[0].set_title('some_digit1')
axes[0].set_xticks(np.arange(0, 28, 3))
axes[0].set_yticks(np.arange(0, 28, 3))
axes[1].imshow(some_digit2.reshape(28, 28), cmap='binary')
axes[1].grid(True)
axes[1].set_title('some_digit2')
axes[1].set_xticks(np.arange(0, 28, 3))
axes[1].set_yticks(np.arange(0, 28, 3))
axes[2].imshow(some_digit3.reshape(28, 28), cmap='binary')
axes[2].grid(True)
axes[2].set_title('some_digit3')
axes[2].set_xticks(np.arange(0, 28, 3))
axes[2].set_yticks(np.arange(0, 28, 3))
plt.show()

# Pre-process the data
# 9. Change the type of y to unit8
y_jungyu = y_jungyu.astype(np.uint8)

# 10. The current target values range from 0 to 9 i.e. 10 classes. Transform the target variable to 3 classes as follows:
  # a. Any digit between 0 and 3 inclusive should be assigned a target value of 0
  # b. Any digit between 4 and 6 inclusive should be assigned a target value of 1
  # c. Any digit between 7 and 9 inclusive should be assigned a target value of 9 
y_transformed = np.where((y_jungyu >= 0) & (y_jungyu <= 3), 0, np.where((y_jungyu >= 4) & (y_jungyu <= 6), 1, 9))

# 11. Print the frequencies of each of the three target classes and note it in your written report in addition provide a screenshot showing a bar chart.
frequencies = pd.Series(y_transformed).value_counts()

some_digits = [0, 1, 9]
digits_count = frequencies[some_digits].tolist()

plt.bar(some_digits, digits_count)
plt.xlabel('Digits')
plt.ylabel('Frequency')
plt.xticks(some_digits)
plt.title('Frequency of Digits')
plt.show()

# 12. Split your data into train, test. Assign the first 50,000 records for training and the last 20,000 records for testing.
X_train, X_test, y_train, y_test = X[:50000], X[50000:], y[:50000], y[50000:]

# Build Classification Models 
# Naïve Bayes
# 13. Train a Naive Bayes classifier using the training data. Name the classifier NB_clf_firstname.
from sklearn.naive_bayes import MultinomialNB
NB_clf_jungyu = MultinomialNB()
NB_clf_jungyu.fit(X_train, y_train)

# 14. Use 3-fold cross validation to validate the training process.
from sklearn.model_selection import cross_val_score
scores = cross_val_score(NB_clf_jungyu, X_train, y_train, cv=3)
print("Cross-validation scores:", scores)

# 15. Use the model to score the accuracy against the test data.
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, NB_clf_jungyu.predict(X_test))

# 16. Generate the accuracy matrix.
confusion_matrix(y_test, NB_clf_jungyu.predict(X_test))

# 17. Use the classifier to predict the three variables you defined in point 7 above.
prediction1_NB = NB_clf_jungyu.predict(some_digit1.reshape(1, -1))
prediction2_NB = NB_clf_jungyu.predict(some_digit2.reshape(1, -1))
prediction3_NB = NB_clf_jungyu.predict(some_digit3.reshape(1, -1))

print(prediction1_NB)
print(prediction2_NB)
print(prediction3_NB)


# Logistic regression (lbfgs)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# 18. Train a Logistic regression classifier using the same training data. Name the classifier LR_clf_firstname. 
# (Note this is a multi-class problem make sure to check all the parameters and set multi_class='multinomial').
# Try training the classifier using two solvers first “lbfgs” then “Saga”. Set max_iter to 1200 and tolerance to 0.1 in both cases.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

LR_clf_lbfgs_jungyu = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1200, tol=0.1)
LR_clf_lbfgs_jungyu.fit(X_train, y_train)

# 19. Use 3-fold cross validation on the training data and note the results in your written response.
cross_val_score(LR_clf_lbfgs_jungyu, X_train, y_train, cv=3)

# 20. Use the model to score the accuracy against the test data
accuracy_score(y_test, LR_clf_lbfgs_jungyu.predict(X_test))

# 21. Generate the accuracy matrix, precision, and recall of the model
confusion_matrix(y_test, LR_clf_lbfgs_jungyu.predict(X_test))
from sklearn.metrics import precision_score, recall_score
precision_score(y_test, LR_clf_lbfgs_jungyu.predict(X_test), average=None)
recall_score(y_test, LR_clf_lbfgs_jungyu.predict(X_test), average=None)

# 22. Use the classifier to predict the three variables defined in point 7 above
prediction1_lbfgs = LR_clf_lbfgs_jungyu.predict(some_digit1.reshape(1, -1))
prediction2_lbfgs = LR_clf_lbfgs_jungyu.predict(some_digit2.reshape(1, -1))
prediction3_lbfgs = LR_clf_lbfgs_jungyu.predict(some_digit3.reshape(1, -1))

print(prediction1_lbfgs)
print(prediction2_lbfgs)
print(prediction3_lbfgs)

# Logistic regression (saga)
# 18. Train a Logistic regression classifier using the same training data.
LR_clf_saga_jungyu = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1200, tol=0.1)
LR_clf_saga_jungyu.fit(X_train, y_train)

# 19. Use 3-fold cross validation on the training data and note the results in your written response.
cross_val_score(LR_clf_saga_jungyu, X_train, y_train, cv=3)

# 20. Use the model to score the accuracy against the test data
accuracy_score(y_test, LR_clf_saga_jungyu.predict(X_test))

# 21. Generate the accuracy matrix, precision, and recall of the model
confusion_matrix(y_test, LR_clf_saga_jungyu.predict(X_test))
precision_score(y_test, LR_clf_saga_jungyu.predict(X_test), average=None)
recall_score(y_test, LR_clf_saga_jungyu.predict(X_test), average=None)

# 22. Use the classifier to predict the three variables defined in point 7 above
prediction1_saga = LR_clf_saga_jungyu.predict(some_digit1.reshape(1, -1))
prediction2_saga = LR_clf_saga_jungyu.predict(some_digit2.reshape(1, -1))
prediction3_saga =LR_clf_saga_jungyu.predict(some_digit3.reshape(1, -1))

print(prediction1_saga)
print(prediction2_saga)
print(prediction3_saga)
