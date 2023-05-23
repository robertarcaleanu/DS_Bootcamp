import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
pd.set_option('display.max_columns', 24)
df = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/kc_house_data.csv")
df.head()
df.info()
df.describe().transpose()

# Check for null values
df.isnull().sum()
sns.heatmap(data=df.isnull())

# EDA
sns.displot(data=df, x='price') # Price distribution
sns.countplot(data=df, x='bedrooms')

sns.heatmap(data=df.drop('id', axis=1).corr(), cmap="BuPu", annot=True)
df.corr()['price'].sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='price', y='sqft_living')

sns.boxplot(data=df, x='bedrooms', y='price')

sns.scatterplot(data=df, x='price', y='long')
sns.scatterplot(data=df, x='price', y='lat')
sns.scatterplot(data=df, x='long', y='lat', hue='price')

# non_top_1_percent = df.sort_values('price', ascending=False).iloc[216:]
sns.scatterplot(data=df[df['price'] < 3500000], x='long', y='lat', hue='price', edgecolor=None, palette='RdYlGn',
                alpha=0.2, s=10)

sns.boxplot(data=df, x='waterfront', y='price')

# Feature engineering
df.head()
# Remove id
df = df.drop('id', axis=1)

# Convert to date
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)

sns.boxplot(data=df, x='month', y='price')
df.groupby('month')['price'].mean().plot()
df.groupby('year')['price'].mean().plot()

df = df.drop('date', axis=1)

# Zip Code
df['zipcode'].value_counts()
# We'll drop the zipcodes, but we could do some feature engineering - group by zones, cheap/expensive
df.drop('zipcode', axis=1, inplace=True)

# Year renovated
df['yr_renovated'].value_counts()
sns.scatterplot(data=df, x='yr_renovated', y='price')
# We'll leave the renovated year since we can have a correlation between year and price
# df.loc[df['yr_renovated'] > 0, 'yr_renovated'] = 1

# Basement - We'll keep it continuous

# Data Preprocessing
# We need numpy arrays when using TF
X = df.drop('price', axis=1).values
y = df['price'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# We need to do a scaling, but after the split because we want to avoid data leakage
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(19, activation='relu')) # 19 neurons as we have 19 features
model.add(tf.keras.layers.Dense(19, activation='relu'))
model.add(tf.keras.layers.Dense(19, activation='relu'))
model.add(tf.keras.layers.Dense(19, activation='relu'))
model.add(tf.keras.layers.Dense(1)) # output


model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          batch_size=128,
          epochs=400)

losses = pd.DataFrame(model.history.history)
losses.plot()

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
