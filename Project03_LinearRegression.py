import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
pd.set_option('display.max_columns', 24)
df = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/Ecommerce Customers.csv")


# Exploratory Data Analysis (EDA)
df.head()
df.info()
df.describe()

sns.jointplot(df, x='Time on Website', y='Yearly Amount Spent')
sns.jointplot(df, x='Time on App', y='Yearly Amount Spent')
sns.jointplot(df,x='Time on App', y='Length of Membership', kind='hex')

sns.pairplot(df)

# Yearly amount spent and length of membership are correlated
sns.lmplot(df, x='Length of Membership', y='Yearly Amount Spent')

# Split the data
X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
Y = df['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)

# Check the coefficients - Length of the membership has the biggest impact
print(lm.coef_)
coefficients = pd.DataFrame(data=lm.coef_, index=X.columns)
coefficients.columns = ['Coeff']

# Predict the test data
predictions = lm.predict(X_test)

# Verify predicted values with actual values - They're correlated; good sign
plt.scatter(Y_test, predictions)
plt.xlabel('Y test')
plt.ylabel('Predictions')
plt.show()
plt.close()

# Evaluate the model
from sklearn import metrics
# Mean Absolute Error
print(metrics.mean_absolute_error(Y_test, predictions))
# Mean squared Error
print(metrics.mean_squared_error(Y_test, predictions))
# Root Mean squared Error
print(np.sqrt(metrics.mean_squared_error(Y_test, predictions)))

# Residuals
plt.hist(Y_test - predictions, bins=50)