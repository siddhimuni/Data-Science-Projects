#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Importing the dataset
# 

# In[2]:


data=pd.read_csv('C:/TDS Data Science/House Data.csv')
data.head()


# In[3]:


data.describe()


# In[4]:


for col in data.columns:
    print (col)


# In[5]:


data['date'].dtypes


# In[6]:


data['date']=pd.to_datetime(data['date'])


# In[7]:


data['year']=data['date'].dt.year
data['year'].head()


# In[8]:


data['month']=data['date'].dt.month
data['month'].head()


# In[9]:


data['day']=data['date'].dt.day
data['day'].head()


# In[10]:


categorical =[var for var in data.columns if data[var].dtype=='O']
print("the categorical variables are: ",categorical)
##implies no categorical variable 


# In[11]:


data.drop(['date'],axis=1,inplace=True)


# In[12]:


for col in data.columns:
    if col!='price':
        print(col)


# In[13]:


count=0
for col in data.columns:
    if col!='price':
          count+=1
print(count)


# In[14]:


data.head()


# In[15]:


data['Price']=data['price']
#data.head()
data.drop(['price'],axis=1,inplace=True)
data.head()


# In[16]:


y=data.iloc[:,-1].values
x=data.iloc[:,:-1].values


# In[17]:


x.shape


# In[18]:


y.shape


# In[19]:


for col in data.columns:
    if col!='price':
        print(col)


# # Checking for linearity
# 

# In[20]:


p = sns.pairplot(data,x_vars=['id','bedrooms','bathrooms','sqft_living','sqft_lot'],y_vars='Price',size=7,aspect=0.7)
#id shows no correlation
#bedroom shows no correlation
#bathroom shows positive correlation
#sqft_living shows positive correlation
#sqft_lot shows no correlation


# In[21]:


data.drop(['id','bedrooms','sqft_lot'],axis=1,inplace=True)


# In[ ]:





# In[22]:


p = sns.pairplot(data,x_vars=['floors','waterfront','view','condition','grade'],y_vars='Price',size=8,aspect=0.8)
#floors shows no correlation
#waterfront shows no correlation
#view shows no correlation
#condition shows no correlation
#grade shows positive correaltion


# In[23]:


data.drop(['floors','waterfront','view','condition'],axis=1,inplace=True)


# In[24]:


data


# In[25]:


for col in data.columns:
    if col!='Price':
        print(col)


# In[26]:


p = sns.pairplot(data,x_vars=['sqft_above','sqft_basement','yr_built','yr_renovated','zipcode'],y_vars='Price',size=8,aspect=0.8)
#sqft_above shows positive correlation
#sqft_basement shows partial positive correlation
#yr_built shows partial positive correlation
#yr_renovates shows no correlation
#zipcode shows partial negative correalation


# In[27]:


data.drop(['yr_renovated'],axis=1,inplace=True)


# In[28]:


p = sns.pairplot(data,x_vars=['lat','long','yr_built','sqft_living15','sqft_lot15'],y_vars='Price',size=8,aspect=0.8)
#lat shows positive correlation
#lot shows negative correalation
#yr_built shows no correaltion
#sqft_living15 shows positive correlation
#sqft_lot15 shows negative correlation


# In[29]:


data.drop(['yr_built'],axis=1,inplace=True)


# In[30]:


data


# In[31]:


p = sns.pairplot(data,x_vars=['year','month','day'],y_vars='Price',size=8,aspect=0.8)
#year,month,day shows no correlation


# In[32]:


data.drop(['year','month','day'],axis=1,inplace=True)


# In[33]:


data


# In[34]:


y=data.iloc[:,-1].values
x=data.iloc[:,:-1].values


# In[35]:


x.shape


# In[36]:


y.shape


# # EDA

# In[37]:


sns.distplot(data['Price'],bins=5)
#price var has positive skewness or outliers to the right


# # Checking Multicollinearity

# In[38]:


sns.pairplot(data)


# In[39]:


sns.heatmap(data.corr(),annot=True)


# In[40]:


plt.scatter(x='sqft_above',y='sqft_living',data=data)


# In[41]:


plt.figure(figsize=(20,10))
sk=dict()
for col in data.columns:
    skewness=data[col].skew()
    print('skewness of {} is {}'.format(col,skewness))
    sk[col]=skewness
sns.barplot(x=list(sk.keys()),y=list(sk.values()),palette='cool')
plt.show()


# In[58]:


data['sqft_lot15'].skew()


# In[53]:


plt.hist(np.log(data['sqft_lot15']))


# In[59]:


sqft_lot15=np.log(data['sqft_lot15'])


# In[60]:


sqft_lot15.skew()


# In[45]:


Price=np.log(data['Price'])


# In[46]:


Price.skew()


# In[50]:


sns.distplot(data['sqft_lot15'],bins=5)


# In[56]:


plt.hist(np.log(data['sqft_lot15']))


# In[88]:


sns.distplot(data['Price'],bins=5)


# In[66]:


sns.distplot(data['sqft_lot15'],bins=5)


# In[61]:


plt.figure(figsize=(20,10))
kr=dict()
for col in data.columns:
    kurtosis=data[col].kurt()
    print('kurtosis of {} is {}'.format(col,kurtosis))
    kr[col]=kurtosis
sns.barplot(x=list(kr.keys()),y=list(kr.values()),palette='cool')
plt.show()


# # Splittig the dataset into training and test set

# In[62]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# # Training the multiple linear regression model on the training set

# In[63]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# # Predicting test set results

# In[66]:


y_pred= regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#smaller values show larger difference whereas large values showb little difference
#this implies that we might have included some useless variable and it is violating one of the three assumptions of OLS


# # Evaluation matrix

# In[73]:


import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_percentage_Error

def regression_results(y_true,y_pred):
    #regression metrics
    explained_variance=metrics.explained_variance_score(y_true,y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true,y_pred)
    mse=metrics.mean_squared_error(y_true,y_pred)
    #mean_squared_log_error=metrics.mean_squared_log_error(y_true,y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true,y_pred)
    r2=metrics.r2_score(y_true,y_pred)
    
    print('explained variance:',round(explained_variance,4))
    #print('mean_squared_log_error',mean_squared_log_error)
    print('mean_squared_error: ',mean_squared_error(y_test,y_pred))
    print('r2: ',round(r2,4))
    print('MAE: ',round(mean_absolute_error,4))
    #print('MAPE: ',round(mean_absolute_percentage_error(y_test,y_pred),4))
    print('Adjusted r2: ',round(1-(1-regressor.score(x_test,y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1),3))
    print('MSE: ',round(mse,4))
    print("RMSE: ",round(np.sqrt(mse),4))
    
    #MAPE+r2=1
    #SST=SSR+SSE(explained variance+MAPE)


# In[74]:


regression_results(y_test,y_pred)


# In[75]:


cdf=pd.DataFrame(regressor.coef_,columns=['Coefficients'])
print(cdf)


# # checking for normality

# In[78]:


residuals = y_test-y_pred
mean_residuals=np.mean(residuals)
print('Mean of Residuals: {}'.format(mean_residuals))
#the scale of data is in lakhs so mean of 8k is accepatable
#hence it is not violating normality assumption


# In[86]:


residuals


# In[79]:


p=sns.scatterplot(y_pred,residuals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-200000,200000)
plt.xlim(-200000,200000)
p=sns.lineplot([-150000,150000],[-150000,150000],color='blue')
p=plt.title('Residuals vs fitted values plot for homoscedasticity')
#there is no fixed pattern in graph hence pattern is homoscedatic in nature


# In[81]:


p=sns.distplot(residuals,kde=True)
p=plt.title('Normality of error terms/residuals')


# In[82]:


regressor.score(x_test,y_test)
#accuracy is 62%


# In[ ]:





# In[ ]:




