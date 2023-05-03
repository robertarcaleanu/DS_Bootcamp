import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

# Load and explore the data
df = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/10-Data"
                 "-Capstone-Projects/911.csv")
df.info()
df.head()

# Questions
# 1. What are the top 5 zipcodes for 911 calls?
df['zip'].value_counts()
# 2. What are the top 5 townships (twp) for 911 calls?
df['twp'].value_counts()
# 3. Take a look at the 'title' column, how many unique title codes are there?
df['title'].nunique()

# Creating new features 1. Obtain reason 2. What is the most common Reason for a 911 call based off of this new
# column? 3. Now use seaborn to create a countplot of 911 calls by Reason 4. Now let us begin to focus on time
# information. What is the data type of the objects in the timeStamp column?
# 5. You should have seen that these
# timestamps are still strings. Use [pd.to_datetime](
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings
# to DateTime objects.
# 6. Create 3 columns called Hour, Month and Day of the Week
