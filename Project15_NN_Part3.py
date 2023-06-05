import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data - We want to predict the status of the load
pd.set_option('display.max_columns', 24)
data_info = pd.read_csv(
    "C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/lending_club_info.csv",
    index_col='LoanStatNew')


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


feat_info('mort_acc')

df = pd.read_csv("C:/Users/rober/OneDrive/Escriptori/DataSets/Python/Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/lending_club_loan_two.csv")
df.head()
df.info()

# EDA ----
# Countplot
sns.countplot(data=df, x='loan_status') # It's imbalanced
plt.close()

sns.histplot(data=df, x='loan_amnt', bins=30)
plt.close()

# Loan distribution
sns.histplot(data=df[df['loan_status'] == 'Fully Paid'], x='loan_amnt', bins=30, label='Fully Paid')
sns.histplot(data=df[df['loan_status'] == 'Charged Off'], x='loan_amnt', bins=30, label='Charged Off')
plt.xlabel('Loan Amount')
plt.ylabel('Count')
plt.legend()
plt.close()

# Correlations
df.corr()
sns.heatmap(data=df.corr(), annot=True) # High correlation between installment and loan_amnt
feat_info('installment')
feat_info('loan_amnt')
sns.scatterplot(data=df, x='installment', y='loan_amnt')
plt.close()

sns.boxplot(data=df, x='loan_status', y='loan_amnt')
plt.close()
df.groupby('loan_status')['loan_amnt'].describe()

# Grade and SubGrade
np.sort(df['grade'].unique())
np.sort(df['sub_grade'].unique())

sns.countplot(data=df, x='grade', hue='loan_status')
plt.close()

sns.countplot(data=df, x='sub_grade', order=np.sort(df['sub_grade'].unique()), palette='coolwarm')
plt.close()
sns.countplot(data=df, x='sub_grade', order=np.sort(df['sub_grade'].unique()), palette='coolwarm', hue='loan_status')
plt.close()

df_FandG = df[(df['grade'] == 'F') | (df['grade'] == 'G')]
sns.countplot(data=df_FandG,
              x='sub_grade', order=np.sort(df_FandG['sub_grade'].unique()), palette='coolwarm', hue='loan_status')

df.loc[df['loan_status'] == 'Fully Paid', 'load_repaid'] = int(1)
df.loc[df['loan_status'] == 'Charged Off', 'load_repaid'] = int(0)
df[['loan_status', 'load_repaid']]

df.corr()['load_repaid'].sort_values()[:-1].plot(kind='bar')

# Data Preprocessing ----
df.head()
df.shape

# Missing values
df.isnull().sum()
df.isnull().sum()/df.shape[0] * 100
sns.heatmap(data=df.isnull())
plt.close()

feat_info('emp_title') # Job titles
df['emp_title'].nunique() # To many - we may drop the column
df['emp_title'].value_counts()
df.drop('emp_title', axis=1, inplace=True)
df.info()

feat_info('emp_length')
df['emp_length'].nunique()
df['emp_length'].value_counts()

emp_length_order = ['< 1 year',
                    '1 year',
                    '2 years',
                    '3 years',
                    '4 years',
                    '5 years',
                    '6 years',
                    '7 years',
                    '8 years',
                    '9 years',
                    '10+ years']

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='emp_length', order=emp_length_order)
plt.close()

sns.countplot(data=df, x='emp_length', order=emp_length_order, hue='loan_status')
plt.close()

df_charged_off = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()['loan_status']
df_paid = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length').count()['loan_status']

df_emp = df_charged_off/df_paid
df_emp.plot(kind='bar')

# Employment length doesn't provide a lot of info - we drop it
df.drop('emp_length', axis=1, inplace=True)
df.isnull().sum()

# Let's compare title and purpose
df[['purpose', 'title']]
df['purpose'].nunique()
df['title'].nunique()

# The title column is simply a string subcategory/description of the purpose column - we drop it
df.drop('title', axis=1, inplace=True)
df.isnull().sum()

feat_info('revol_util')
feat_info('mort_acc')
feat_info('pub_rec_bankruptcies')

df['mort_acc'].value_counts() # we need to fill values - check correlation with other variables

df.corr()['mort_acc'].sort_values()
feat_info('total_acc')
df['total_acc'].value_counts()

# Let's try this fillna() approach. We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
total_acc_avg[2.0]


def fill_mort_acc(total_acc, mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.

    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1) # Axis = 1 -> Columns
df.isnull().sum().sort_values()

# revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data
df = df.dropna()

# Categorical Variables and Dummy Variables ----
df.select_dtypes(exclude='number').columns

df[df.select_dtypes(exclude='number').columns].head()

# Term

# Grade

# Subgrade


# Home_ownership

# verification_status, application_type,initial_list_status,purpose

# Address

# Earliest_cr_line


# Train test split ---



# Evaluating Model Performance ----