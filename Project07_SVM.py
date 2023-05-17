import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
pd.set_option('display.max_columns', 24)
iris = sns.load_dataset('iris')

# EDA
iris.head()
iris.info()
iris.describe()

# Setosa can be identified easily
sns.pairplot(data=iris, hue="species")
sns.kdeplot(data=iris[iris["species"] == "setosa"], x="sepal_width", y="sepal_length", cmap="plasma",
            shade_lowest=False, shade=True)
sns.countplot(data=iris, x="species")

# Train test split
from sklearn.model_selection import train_test_split

X = iris.drop("species", axis=1)
Y = iris["species"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

# Gridsearh - we use gridsearch for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid=param_grid, refit=True, verbose=3)

grid.fit(X_train, Y_train)
# Print the best parameters
print(grid.best_params_)

# Re-run model with new parameters
grid_predictions = grid.predict(X_test)
print(confusion_matrix(Y_test, grid_predictions))
print(classification_report(Y_test, grid_predictions))

