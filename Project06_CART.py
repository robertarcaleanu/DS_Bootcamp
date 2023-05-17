import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
pd.set_option('display.max_columns', 24)
loans = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/loan_data.csv")

# EDA
loans.head()
loans.info()
loans.describe()

plt.figure(figsize=(10, 6))
sns.histplot(data=loans, x="fico", hue="credit.policy", bins=30)
sns.histplot(data=loans, x="fico", hue="not.fully.paid", bins=30)

sns.countplot(data=loans, x="purpose", hue="not.fully.paid")

sns.jointplot(data=loans, x="fico", y="int.rate", color="purple")

# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
# sns.lmplot(data=loans, x="fico", y="int.rate", hue="credit.policy", )
# sns.lmplot(data=loans, x="fico", y="int.rate", hue="credit.policy", ax=ax2)

sns.lmplot(y='int.rate', x='fico', data=loans, hue='credit.policy',
           col='not.fully.paid', palette='Set1')
# sns.pairplot(data=loans, hue="not.fully.paid")

# Setting up the data
loans.info()

# Categorical Features - purpose
# We perform one-hot encoding for categorical features
final_data = pd.get_dummies(loans, columns=["purpose"], drop_first=True)
final_data.info()

# Train test split
from sklearn.model_selection import train_test_split

X = final_data.drop("not.fully.paid", axis=1)
Y = final_data["not.fully.paid"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Train a decision tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)

# Predict with decision tree
dtree_predict = dtree.predict(X_test)

# Model evaluation
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, dtree_predict))
print(classification_report(Y_test, dtree_predict))

# WIP
# plot_tree(dtree)

# Train Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=600)
rf.fit(X_train, Y_train)

rf_predict = rf.predict(X_test)
print(confusion_matrix(Y_test, rf_predict))
print(classification_report(Y_test, rf_predict))

# Note that the target is not balanced, so we need to pay more attention to recall than overall accuracy
final_data["not.fully.paid"].value_counts()