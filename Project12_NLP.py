import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
pd.set_option('display.max_columns', 24)
yelp = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/yelp.csv")

yelp.head()
yelp.info()
yelp.describe()

yelp["text length"] = yelp["text"].apply(len)

# EDA
g = sns.FacetGrid(yelp, col="stars")
g = g.map(plt.hist, "text length")

sns.boxplot(data=yelp, y="text length", x="stars")
sns.countplot(data=yelp, x="stars")

yelp.groupby("stars").mean()
sns.heatmap(yelp.groupby("stars").mean().corr(), annot=True)

# NLP Task
# We're going to use only 1 and 5 stars reviews
yelp_class = yelp[(yelp["stars"] == 1) | (yelp["stars"] == 5)]
yelp_class.groupby("stars").count()

X = yelp_class["text"]
Y = yelp_class["stars"]

from sklearn.feature_extraction.text import CountVectorizer
X = CountVectorizer().fit_transform(X) # This is a sparse matrix word X Text
print(X)

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Train the model
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

# Let's use TF-IDF (term frequency-inverse document frequency) to the process and leverage the pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("bow", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("classifier", MultinomialNB()),
])

# Split the data again
X = yelp_class["text"]
Y = yelp_class["stars"]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Use the pipeline to predict
pipeline.fit(X_train, Y_train)
predictions = pipeline.predict(X_test)

print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))


# Improvement
# Create a text processor

def text_process(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    ...


cv = CountVectorizer(analyzer=text_process)