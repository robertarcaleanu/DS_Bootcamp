import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load the data
pd.set_option('display.max_columns', 24)
df = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data.csv")

# EDA
df.head()
df.describe()
df.info()

sns.pairplot(data=df, hue="TARGET CLASS")

# Standardise the data
X = df.columns[:-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[X])
scaled_features = scaler.transform(df[X])

df_features = pd.DataFrame(scaled_features, columns=X)

# Test and train split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_features, df["TARGET CLASS"], test_size=0.3, random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)

# Predictions and evaluations
prediction = knn.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, prediction))
print(classification_report(Y_test, prediction))

# Elbow method
error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# KNN with K = 25
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
print(confusion_matrix(Y_test, prediction))
print(classification_report(Y_test, prediction))