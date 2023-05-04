import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# Creating new features
# 1. Obtain reason
def split_title(x):
    res = x.split(':')
    return res[0]


df['Reason'] = df['title'].apply(split_title)

# 2. What is the most common Reason for a 911 call based off of this new column?
df['Reason'].value_counts()

# 3. Now use seaborn to create a countplot of 911 calls by Reason
sns.countplot(x=df['Reason'])
plt.show()
plt.close()

# 4. Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?
type(df['timeStamp'].iloc[0])

# 5. You should have seen that these
# timestamps are still strings. Use [pd.to_datetime](
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings
# to DateTime objects.
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

# 6. Create 3 columns called Hour, Month and Day of the Week
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)

# 7. Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:
dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)

# 8.  Now use seaborn to create a countplot
sns.countplot(x='Day of Week', data=df, hue='Reason', palette='viridis')
plt.show()
plt.close()
sns.countplot(x='Month', data=df, hue='Reason', palette='viridis')
plt.show()
plt.close()

# 9. Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation.
byMonth = df.groupby('Month').count()
byMonth.head()

# 10. Now create a simple plot off of the dataframe indicating the count of calls per month.
byMonth['twp'].plot()

# 11. Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column.


# 12. Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method.

# 13. Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.

# 14. Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call

# 15. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week.

# 16. Now create a HeatMap using this new DataFrame.

# 17. Now create a clustermap using this DataFrame.

# 18. Now repeat these same plots and operations, for a DataFrame that shows the Month as the column.

