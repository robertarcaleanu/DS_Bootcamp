import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
pd.set_option('display.max_columns', 24)
df = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/17-K-Means-Clustering/College_Data.csv")

# EDA
df.head()
df.info()
df.describe()

sns.scatterplot(data=df, x="Room.Board", y="Grad.Rate", hue="Private")
sns.scatterplot(data=df, x="Outstate", y="F.Undergrad", hue="Private")

sns.histplot(data=df, x="Outstate", hue="Private")
sns.histplot(data=df, x="Grad.Rate", hue="Private")

# We have a grad rate > 100%
df[df["Grad.Rate"] > 100]
# We re-assign the value to 100
df.loc[df["Grad.Rate"] > 100, "Grad.Rate"] = 100
sns.histplot(data=df, x="Grad.Rate", hue="Private")

# K-means cluster
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2)
km.fit(df.drop("Private", axis=1))
prediction = km.labels_

# Improvements - WIP
# 1. Standardize the data
# 2. Elbow method
# from yellowbrick.cluster import KElbowVisualizer
# visualizer = KElbowVisualizer(km, k=(2, 12))
# visualizer.fit(data)
# visualizer.show()


