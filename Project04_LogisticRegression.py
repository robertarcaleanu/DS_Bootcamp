import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 24)
ad_data = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/advertising.csv")

# EDA
ad_data.head()
ad_data.info()
ad_data.describe()

sns.histplot(ad_data['Age'], bins=35)
sns.jointplot(data=ad_data, x='Age', y='Area Income')
sns.jointplot(data=ad_data, x='Age', y='Daily Time Spent on Site', kind='kde', color='red')
sns.jointplot(data=ad_data, x='Daily Time Spent on Site', y='Daily Internet Usage')
sns.pairplot(data=ad_data, hue='Clicked on Ad', palette='bwr')
sns.countplot(data=ad_data, hue='Male', x='Clicked on Ad', palette='bwr')

# We can use the heatmap to check NA values - we don't have
sns.heatmap(ad_data.isnull(), cbar=False)

# Logistic Regression - Features and target
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

# Test and Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train the model
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, Y_train)

# Predictions and evaluations
predictions = log_model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))