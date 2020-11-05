#%%
import pandas as pd

# read the csv file
df = pd.read_csv('weight-height.csv')

# print the first 5 rows of the data set
df.head()
# %%
# shape of the dataframe
df.shape
#%%
# data type of each column
df.dtypes
#%%
# number of null values
df.info()
# %%
# number of unique values of column Gender
df.Gender.nunique()
# %%
# unique values of column Gender
df.Gender.unique()
# %%
