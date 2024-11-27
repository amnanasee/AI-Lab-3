#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np


# In[12]:


import pandas as pd


# In[5]:


from sklearn.datasets import load_iris


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


from sklearn.preprocessing import MinMaxScaler


# In[20]:


from sklearn.preprocessing import StandardScaler


# In[17]:


iris = load_iris


# In[33]:


X = pd.DataFrame


# In[37]:


iris = load_iris()


# In[39]:


print(iris.DESCR)


# In[43]:


X = pd.DataFrame(iris.data)


# In[45]:


Y = iris.target


# In[47]:


X.head


# In[49]:


X.shape


# In[51]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)


# In[53]:


print(Y)


# In[55]:


X_train_n = X_train.copy()


# In[57]:


X_test_n = X_test.copy()


# In[59]:


norm = MinMaxScaler().fit(X_train_n)


# In[63]:


X_train_norm = norm.transform(X_train_n)


# In[65]:


X_test_norm = norm.transform(X_test_n)


# In[67]:


X_train_norm_df = pd.DataFrame(X_test_norm)


# In[69]:


X_train_norm_df.columns = iris.feature_names


# In[71]:


X_train_norm_df.describe()


# In[73]:


print(X_train_norm_df)


# In[22]:


df = pd.read_csv(r"C:\Users\WELCOME\Desktop\heart.csv")


# In[87]:


print(df)


# In[91]:


df.head()


# In[93]:


df.shape


# In[24]:


X = df
Y = df.target


# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)


# In[28]:


X_train_n = X_train.copy()


# In[30]:


X_test_n = X_test.copy()


# In[32]:


norm = MinMaxScaler().fit(X_train_n)


# In[34]:


X_train_norm = norm.transform(X_train_n)


# In[36]:


X_test_norm = norm.transform(X_test_n)


# In[38]:


X_train_norm_df = pd.DataFrame(X_train_norm)


# In[117]:


X_train_norm_df.describe()


# In[119]:


print(X_train_norm_df)


# In[42]:


num_output_categories = len(Y.unique())
print("Number of Output Categories:", num_output_categories)
print(Y.unique())


# In[44]:


num_data_rows = df.shape[0]
print("Number of Data Rows:", num_data_rows)


# In[46]:


num_training_rows = X_train.shape[0]
print("Number of Training Rows:", num_training_rows)


# In[48]:


num_testing_rows = X_test.shape[0]
print("Number of Testing Rows:", num_testing_rows)


# In[ ]:




