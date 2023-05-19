import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load data
pd.set_option('display.max_columns', 24)
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/19-Recommender-Systems/u.data",
                 sep='\t', names=column_names)
movie_titles = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/19-Recommender-Systems/Movie_Id_Titles")

# EDA
df.head()
df.info()
df.describe()
movie_titles.head()

df = pd.merge(df, movie_titles, on="item_id")
df.head()

# Best ratings
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = df.groupby('title')['rating'].count()

sns.histplot(data=ratings, x="num of ratings", bins=70)
sns.boxplot(data=df, y="rating")

sns.histplot(data=ratings, x="rating")
sns.jointplot(data=ratings, x="rating", y="num of ratings")

# Recommender system - Item Based
# We need a matrix
movie_mat = df.pivot_table(index="user_id", columns="title", values="rating")
movie_mat.head()

# Select two movies that we want to correlate with
ratings.sort_values("num of ratings", ascending=False)

# Let's use Star Wars and Liar Liar
starwars_user_ratings = movie_mat['Star Wars (1977)']
liarliar_user_ratings = movie_mat['Liar Liar (1997)']

# Let's see the movies that correlate with StarWars and Liar Liar
similar_to_starwars = movie_mat.corrwith(starwars_user_ratings)
similar_to_liarliar = movie_mat.corrwith(liarliar_user_ratings)

# Let's create a dataframe to analyse the correlations
corr_starwars = pd.DataFrame(similar_to_starwars, columns=["Correlation"])
corr_starwars.head()
corr_starwars.info()
corr_starwars.dropna(inplace=True)
corr_starwars.info()

# Let's include the number of ratings - note we use join as we use the index to merge the data
corr_starwars = corr_starwars.join(ratings["num of ratings"])

# We'll consider films that have more than 100 reviews
corr_starwars[corr_starwars["num of ratings"] > 100].sort_values("Correlation", ascending=False).head(10)


# Improvement: create a function that will give the best recommendations based on the title
def get_recommendations(title):
    ...
