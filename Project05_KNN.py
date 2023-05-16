import pandas as pd
import seaborn as sns

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