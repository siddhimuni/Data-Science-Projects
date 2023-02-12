#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('C:/Users/user/Downloads/car_prediction_data+Code/car_prediction_data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


categorical = ['Fuel_Type','Seller_Type','Transmission','Owner']


# In[6]:


for ele in categorical:
    print(df[ele].unique())


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df = df.drop(['Car_Name'],axis=1)


# In[10]:


print(df.head())


# In[11]:


df['Current_year']=2021


# In[12]:


df.head()


# In[13]:


df['no_of_years']=df['Current_year']-df['Year']


# In[14]:


df.head()


# In[15]:


df.drop(['Year','Current_year'],inplace=True,axis=1)


# In[16]:


df.head()


# In[17]:


df = pd.get_dummies(df,drop_first=True)


# In[18]:


df.head()


# In[19]:


df.corr()


# In[20]:


import seaborn as sns


# 

# In[21]:


sns.pairplot(df)


# In[22]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# In[23]:


df.head()


# In[24]:


y = df.iloc[:,0].values
x = df.iloc[:,1:].values


# In[25]:


x


# In[26]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)


# In[27]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test , y_train,y_test = tts(x,y,test_size=0.3,random_state=0)


# # Using random forest regressor

# In[28]:


from sklearn.ensemble import RandomForestRegressor


# In[29]:


rf_regressor = RandomForestRegressor()


# In[30]:


from sklearn.model_selection import RandomizedSearchCV


# In[31]:


n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(5,30,num=6)]
min_samples_split =[2,5,10,15,100]
min_samples_leaf = [1,2,5,10]


# In[32]:


random_grid ={'n_estimators' : n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf':min_samples_leaf}
print(random_grid)


# In[33]:


rf = RandomForestRegressor()


# In[34]:


rf = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,random_state=42,n_jobs=1)


# In[35]:


rf.fit(x_train,y_train)


# In[36]:


rf.best_params_


# In[37]:


rf.best_score_


# In[38]:


pred = rf.predict(x_test)


# In[39]:


sns.distplot(y_test-pred)


# In[40]:


plt.scatter(y_test,pred)


# In[41]:



np.set_printoptions(precision=2)

print(np.concatenate((pred.reshape(len(pred),1),y_test.reshape(len(y_test),1)),1))


# In[42]:


from sklearn import metrics


# In[43]:


print('MAE: ', metrics.mean_absolute_error(y_test,pred))
print('MSE: ',metrics.mean_squared_error(y_test,pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,pred)))


# # Using XGBoost regressor

# In[47]:


get_ipython().system('pip install xgboost')


# In[55]:


import  xgboost as xgb
from scipy.stats import uniform,randint


# In[56]:


xgb_model = xgb.XGBRegressor(objective="reg:linear",random_state=42)


# In[57]:


param = {
    "gamma":uniform(0,0.5),
    "learning_rate":uniform(0.003,0.3),
    "max_depth":randint(2,6),
    "n_estimators":randint(100,150),
    "subsample":uniform(0.6,0.4)
}


# In[66]:


xgb = RandomizedSearchCV(estimator= xgb_model, param_distributions=param,n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[67]:


xgb.fit(x_train,y_train)


# In[68]:


xgb.best_score_


# In[69]:


xgb.best_params_


# In[70]:


pred=xgb.predict(x_test)


# In[71]:


sns.distplot(y_test-pred)


# In[72]:


plt.scatter(y_test,pred)


# In[73]:


np.set_printoptions(precision=2)

print(np.concatenate((pred.reshape(len(pred),1),y_test.reshape(len(y_test),1)),1))


# In[74]:


print('MAE: ', metrics.mean_absolute_error(y_test,pred))
print('MSE: ',metrics.mean_squared_error(y_test,pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,pred)))


# # Using LGBM regressor

# In[77]:


get_ipython().system('pip install lightgbm')


# In[78]:


from lightgbm import LGBMRegressor


# In[79]:


lb=LGBMRegressor()


# In[80]:


lb.fit(x_train,y_train)


# In[81]:


params={
    "gamma":uniform(0,0.5),
    "learning_rate":uniform(0.03,0.3),
    "max_depth":randint(2,6),
    "n_estimators":randint(100,150),
    "subsample":uniform(0.6,0.4)
}


# In[82]:


lb = RandomizedSearchCV(estimator=lb,param_distributions=params,n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[83]:


lb.fit(x_train,y_train)


# In[84]:


lb.best_score_


# In[85]:


lb.best_params_


# In[87]:


pred = lb.predict(x_test)


# In[88]:


sns.distplot(y_test-pred)


# In[89]:


plt.scatter(y_test,pred)


# In[90]:


np.set_printoptions(precision=2)

print(np.concatenate((pred.reshape(len(pred),1),y_test.reshape(len(y_test),1)),1))


# In[91]:


print('MAE: ', metrics.mean_absolute_error(y_test,pred))
print('MSE: ',metrics.mean_squared_error(y_test,pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,pred)))


# In[ ]:




