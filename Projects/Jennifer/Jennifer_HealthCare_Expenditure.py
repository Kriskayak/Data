import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv("/Users/jennifer/Jen_Data_Science/healthcare-expenditure-vs-gdp.csv", sep=',')

type(df)

len(df)

data= pd.read_csv('/Users/jennifer/Jen_Data_Science/healthcare-expenditure-vs-gdp.csv', sep=',',
                  usecols=[0,1,2,3,4],
                  skiprows=[0],header=0,names=['Country','Code','Year','GDP per Capita','Healthcare Expenditure per Capita'])

df.shape

# Over here we see the columns names and their data types
df.dtypes

# This provides some statistics on the numerical data
df.describe()


# ## Dealing with missing values
# 
# With every dataset it is vital to evaluate the missing values. How many are there? Is it an error? Are there too many missing values? Does a missing value have a meaning relative to its context?
# We can sum up the total missing values using the following:
#isna-> is not a number (shows if true or false)

df.isna()

#Dealing with missing values? How many np.nan per column?

df.isna().sum() 


# Now that we have identified our missing values, we have a few options. We can fill them in with a certain value (zero, mean/max/median by column, string) or drop them by row. Since there are few missing values, we can drop the rows to avoid skewing the data in further analysis.
df = df.dropna(axis = 0, how = 'any')  #drop rows with any column having np.nan values

#Rename columns
#df.rename(index =str, columns = {'patient_id':'patient_id'})
df


# This allows us to drop rows with any missing values in them.
# 
# ## Inspecting duplicates
# To view repeating rows we can start off by looking at the number of unique values in each column.

# In[13]:


# Here we list all columns
df.columns


# In[14]:


# the number changed cuz we dropped all the isna's
len(df)


# In[15]:


# Its good to inspect your unique key identifier
df.nunique()


# We see here that although there are 690 rows, there are only 637 unique patient_id’s. This could mean that some patient appear more than once in the dataset. To isolate these patients and view their data, we use the following:

# In[16]:


# This shows rows that show up more than once and have the exact same column values. 
# We are keeping the last value for each patient
df[df.duplicated(keep = 'last')]


# In[17]:


# # This shows all instances where patient_id shows up more than once, but may have varying column values
df[df.duplicated(subset = 'patient_id', keep =False)].sort_values('patient_id')


# In[18]:


#Now that I have seen that there are some duplicates, I am going to go ahead and remove any duplicate rows

#df = df.drop_duplicates(subset = None, keep ='first')


# In[19]:


repeat_patients = df.groupby(by = 'patient_id').size().sort_values(ascending =False)
repeat_patients


# This shows that one patient shows up in the data 6 times!
# 
# ## Filtering data
# If we want to remove patients that show up more that 2 times in the data set.

# In[20]:


filtered_patients = repeat_patients[repeat_patients > 2].to_frame().reset_index()
filtered_df = df[~df.patient_id.isin(filtered_patients.patient_id)]
filtered_df


# If we did not have the tilde (“~”) we would get all individuals that repeat more than twice. By adding a tilde the pandas boolean series is reversed and thus the resulting data frame is of those that do NOT repeat more than twice.
# 
# ## Reshaping data
# The dataset has elements of categorical data in the “doctor_name” column. To feed this data into a machine learning pipeline, we will need to convert it into a one hot encoded column. This can be done with a sci-kit learn package, however we will do it in pandas to demonstrate the pivoting and merging functionality. Start off by creating a new dataframe with the categorical data.

# In[63]:


categorical_df = df[['patient_id','doctor_name']]
categorical_df


# In[65]:


# This specifies all rows (':') and column name 'doctor_count'
categorical_df.loc[:,'doctor_count'] = 1


# In[66]:


categorical_df


# We add a column an extra column to identify which doctor a patient deals with. Pivot this table so that we only have numerical values in the cells and the columns become the doctors’ name. Then fill in the empty cells with 0.

# In[67]:


# hot-encoding is assigning numbers to categorical or subjective data (strings) for stats.
doctors_one_hot_encoded = pd.pivot_table( categorical_df,
                                  index = categorical_df.index, 
                                  columns = ['doctor_name'], 
                                   values = ['doctor_count'] )
doctors_one_hot_encoded = doctors_one_hot_encoded.fillna(0)
doctors_one_hot_encoded


# Then drop the multiIndex columns:
# 

# In[68]:


doctors_one_hot_encoded.columns = doctors_one_hot_encoded.columns.droplevel()
doctors_one_hot_encoded


# We can now join this back to our main table. Typically a left join in pandas looks like this:
# 
# `leftJoin_df = pd.merge(df1, df2, on ='col_name', how='left')`
# 
# However we are joining on the index so we pass the “left_index” and “right_index” option to specify that the join key is the index of both tables
# 
# 

# In[69]:


combined_df = pd.merge(df, doctors_one_hot_encoded, left_index = True,right_index =True, how ='left')
combined_df


# We can drop the column that we no longer need by the following:

# In[70]:


combined_df = combined_df.drop(columns=['doctor_name'])
combined_df


# ## Row-wise Operations
# Another key component in data wrangling is having the ability to conduct row-wise or column wise operations. Examples of this are; rename elements within a column based on its value and create a new column that yields a specific value based on multiple attributes within the row.
# 
# For this example lets create a new column that categorizes a patients cell as normal or abnormal based on its attributes. We first define our function and the operation that it will be doing.

# In[71]:


def celltypelabel(x):
    if ((x['cell_size_uniformity'] < 5) &      (x['cell_shape_uniformity'] < 5)):
        
        return('normal')
    else:
        return('abnormal')


# Then we use the pandas apply function to run the celltypelabel(x) function on the dataframe.

# In[72]:


combined_df['cell_type_label'] = combined_df.apply(lambda x: celltypelabel(x), axis=1)


# In[73]:


combined_df


# ## Conclusion
# Although some of these data manipulation steps can be done in SAS and excel. Doing it in python not only allows you to connect the data to vast open source resources in computer vision, machine and deep learning, but also for ETL automation purposes and more.

# # Homework
# 1. Hot encode a new column in this dataset for cancerous (1) or not cancerous (0).

# In[79]:


categorical_df = df[['patient_id','class']]
categorical_df


# In[84]:


categorical_df.loc[:,'class_type'] = 1
categorical_df


# In[78]:


malignancy_one_hot_encoded = pd.pivot_table( categorical_df,
                                  index = categorical_df.index, 
                                  columns = ['class'], 
                                   values = ['class_count'] )
malignancy_one_hot_encoded = malignancy_one_hot_encoded.fillna(0)
malignancy_one_hot_encoded


# In[85]:


malignancy_one_hot_encoded.columns = malignancy_one_hot_encoded.columns.droplevel()
malignancy_one_hot_encoded


# In[86]:


combined_df = pd.merge(df, malignancy_one_hot_encoded, left_index = True,right_index =True, how ='left')
combined_df


# In[87]:


combined_df = combined_df.drop(columns=['class'])
combined_df


# In[ ]:




