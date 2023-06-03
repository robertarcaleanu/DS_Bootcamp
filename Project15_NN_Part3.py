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

# EDA
# Countplot
sns.countplot(data=df, x='loan_status') # It's imbalanced
sns.histplot(data=df, x='loan_amnt', bins=30)

# Loan distribution
sns.histplot(data=df[df['loan_status'] == 'Fully Paid'], x='loan_amnt', bins=30, label='Fully Paid')
sns.histplot(data=df[df['loan_status'] == 'Charged Off'], x='loan_amnt', bins=30, label='Charged Off')
plt.xlabel('Loan Amount')
plt.ylabel('Count')
plt.legend()

# Correlations
df.corr()
sns.heatmap(data=df.corr(), annot=True) # High correlation between installment and loan_amnt
feat_info('installment')
feat_info('loan_amnt')
sns.scatterplot(data=df, x='installment', y='loan_amnt')

sns.boxplot(data=df, x='loan_status', y='loan_amnt')
df.groupby('loan_status')['loan_amnt'].describe()

# Grade and SubGrade
np.sort(df['grade'].unique())
np.sort(df['sub_grade'].unique())

sns.countplot(data=df, x='grade', hue='loan_status')

sns.countplot(data=df, x='sub_grade', order=np.sort(df['sub_grade'].unique()), palette='coolwarm')
sns.countplot(data=df, x='sub_grade', order=np.sort(df['sub_grade'].unique()), palette='coolwarm', hue='loan_status')

df_FandG = df[(df['grade'] == 'F') | (df['grade'] == 'G')]
sns.countplot(data=df_FandG,
              x='sub_grade', order=np.sort(df_FandG['sub_grade'].unique()), palette='coolwarm', hue='loan_status')

df.loc[df['loan_status'] == 'Fully Paid', 'load_repaid'] = int(1)
df.loc[df['loan_status'] == 'Charged Off', 'load_repaid'] = int(0)
df[['loan_status', 'load_repaid']]

df.corr()['load_repaid'].sort_values()[:-1].plot(kind='bar')

# Data Preprocessing

