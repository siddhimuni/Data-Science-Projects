#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


data=pd.read_csv('C:\data.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


data['country'].unique()


# In[7]:


data['guests']=data['adults']+data['children']+data['babies']


# In[8]:


data.head()


# In[9]:


col=['adults','children','babies']
data.drop(columns=col,inplace=True,axis=1)


# In[10]:


data.head()


# In[11]:


data['guests'].value_counts()


# In[17]:


data = data[data['guests']!=0]


# In[18]:


data['guests'].value_counts()


# In[19]:


data.isnull().sum()


# In[20]:


data['country'].value_counts()


# In[23]:


data['country'].fillna(data['country'].mode()[0],inplace=True)


# In[24]:


data.isnull().sum()


# In[25]:


data['agent'].unique()


# In[27]:


data['agent'].fillna(data['agent'].median(),inplace=True)


# In[28]:


data.isnull().sum()


# In[30]:


data['company'].value_counts()


# In[32]:


data['company'].fillna(data['company'].median(),inplace=True)


# In[33]:


data.isnull().sum()


# In[34]:


data['guests'].value_counts()


# In[35]:


data['guests'].unique()


# In[36]:


sns.distplot(data['guests'])


# In[37]:


data['guests'].fillna(data['guests'].median(),inplace=True)


# In[38]:


data.isnull().sum()


# In[39]:


data.head()


# In[40]:


for ele in data.columns:
    print(ele)


# In[42]:


data['is_repeated_guest'].value_counts()


# In[49]:


data['is_canceled']


# In[50]:


pl = sns.countplot(x='is_canceled',hue='is_repeated_guest')
labels=('check out','cancelled')
pl.set_xticklabels(labels)
LAB={'Repeated Guest','First time guest'}
plt.legend(labels=LAB)


# In[53]:


plt.figure(figsize=(15,10))
sns.countplot(x='arrival_date_month',data=data,order=pd.value_counts(data['arrival_date_month']).index)
plt.xlabel('Month')
plt.ylabel('No of bookings')


# In[54]:


plt.figure(figsize=(15,10))
sns.boxplot(x='customer_type',y='stays_in_weekend_nights',data=data)


# In[55]:


plt.figure(figsize=(15,10))
sns.boxplot(x='customer_type',y='stays_in_weekend_nights',data=data)


# In[ ]:




