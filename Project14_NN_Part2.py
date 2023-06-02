import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data - Regression exercise
pd.set_option('display.max_columns', 24)
df = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/cancer_classification.csv")
df.head()
df.describe().transpose()

# EDA
sns.countplot(data=df, x='benign_0__mal_1') # To check if it's balanced
# Check correlations
sns.heatmap(df.corr(), annot=True)
df.corr()['benign_0__mal_1'].sort_values()[:-1].plot(kind='bar')

# Train test split
X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(15, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # it's binary classification

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train, y_train, epochs=600, validation_data=(X_test, y_test))

losses = pd.DataFrame(model.history.history)
losses.plot() # We can see that we're overfitting

# HOW TO AVOID OVERFITTING
# Let's use early stopping
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dense(15, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # it's a binary classification

model.compile(loss='binary_crossentropy', optimizer='adam')

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25) # We want to minimise the validation loss
model.fit(X_train, y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

#Let's try dropout layers
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5)) # Percentage of dropout neurons - only 50% will be turned on // rate: Float between 0 and 1. Fraction of the input units to drop.
model.add(tf.keras.layers.Dense(15, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # it's a binary classification

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train, y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

model_loss_dropout = pd.DataFrame(model.history.history)
model_loss_dropout.plot()

# Predict values
predictions = (model.predict(X_test) > 0.5).astype('int32')

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))