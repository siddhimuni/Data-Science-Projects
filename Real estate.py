#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('C:/Project datasets/data_house.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data['CHAS'].value_counts()


# In[7]:


data.describe()


# In[8]:



data.hist(bins=50,figsize=(20,15))
plt.show()


# In[9]:


x=data.iloc[:,:-1].values


# In[10]:


x


# In[11]:


y=data.iloc[:,-1].values


# In[12]:


y


# In[13]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2,random_state=0)


# In[14]:


x_train.shape


# In[15]:


x_test.shape


# In[16]:


train_set,test_set=tts(data,test_size=0.2,random_state=0)


# In[17]:


train_set.shape


# In[18]:


test_set.shape


# In[19]:


train_set.head()


# In[20]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=0)
for x,y in split.split(data,data['CHAS']):
    strat_train_set = data.loc[x]
    strat_test_set = data.loc[y]


# In[21]:


strat_train_set.head()


# In[22]:


strat_test_set.head()


# In[23]:


strat_test_set['CHAS'].value_counts()


# In[25]:


strat_train_set['CHAS'].value_counts()


# In[26]:


95/7


# In[27]:


376/28


# In[28]:


data = strat_train_set.copy()


# In[29]:


corr_matrix = data.corr()


# In[30]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[31]:


from pandas.plotting import scatter_matrix
attributes =['MEDV','RM','ZN','LSTAT']
scatter_matrix(data[attributes],figsize=(12,8))


# In[32]:


data.plot(kind='scatter',x='RM',y='MEDV')


# In[33]:


data['TAX_RM'] = data['TAX']/data['RM']


# In[34]:


data.head()


# In[35]:


corr_matrix = data.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[36]:


data.plot(kind='scatter',x='TAX_RM',y='MEDV')


# In[37]:


for ele in data.columns:
    data[ele].fillna(data[ele].median(),inplace=True)


# In[38]:


data.describe()


# Feature Scaling
# 1. Normalization
#    (value-min)/(max-min)
#     class used - MinMaxScaler
# 2. Standardization
#    (value-mean)/std
#    class used - StandardScaler

# # Creating a Pipeline

# In[39]:


from sklearn.impute import SimpleImputer


# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(),

my_pipeline = Pipeline([
    
    ('std_scaler',StandardScaler()),
])


# In[41]:


data.describe()


# In[42]:


data_tr = my_pipeline.fit_transform(data)


# In[43]:


data_tr


# # Selecting a desired model

# In[44]:


data_tr.shape


# In[45]:


data = strat_train_set.drop('MEDV',axis=1)
data_labels = strat_train_set['MEDV'].copy()


# In[46]:


data


# In[60]:


#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(data,data_labels)


# In[61]:


some_data = data.iloc[:5]


# In[62]:


some_data


# In[63]:


some_labels= data_labels.iloc[:5]


# In[64]:


some_labels


# In[65]:


model.predict(some_data)


# # Evaluating the model

# In[66]:


from sklearn.metrics import mean_squared_error
data_predict = model.predict(data)


# In[67]:


lin_mse = mean_squared_error(data_labels,data_predict)


# In[68]:


lin_rmse = np.sqrt(lin_mse)


# In[69]:


lin_rmse


# # Using better evaluation technique - Cross Validation

# In[70]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,data_tr,data_labels,scoring='neg_mean_squared_error',cv=10)


# In[71]:


rmse_scores = np.sqrt(-scores)


# In[72]:


rmse_scores


# In[73]:


def print_scores(scores):
    print('Scores',scores)
    print('Mean', scores.mean())
    print('Standard deviation:',scores.std())


# In[74]:


print_scores(rmse_scores)


# # Saving the model

# In[75]:


from joblib import dump,load
dump(model, 'Dragon.joblib')


# # Testing the model on test set

# In[78]:


strat_test_set


# In[79]:


x_test = strat_test_set.drop('MEDV',axis=1)


# In[80]:


y_test = strat_test_set['MEDV'].copy()


# In[81]:


pred = model.predict(x_test)


# In[82]:


y_test


# In[92]:


temp = np.array(x_test)


# In[95]:


temp_pred = np.array(y_test)


# In[96]:


temp_pred[0]


# In[93]:


temp


# In[94]:


temp[0]


# In[83]:


pred


# In[84]:


final_mse = mean_squared_error(y_test,pred)


# In[85]:


final_rms = np.sqrt(final_mse)


# In[86]:


final_rms


# In[ ]:




