#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('C:/Train.csv')


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0])


# In[7]:


data.isnull().sum()


# In[8]:


data['Item_Weight']=data['Item_Weight'].fillna(data['Item_Weight'].median())


# In[9]:


data.isnull().sum()


# In[10]:


cat=[]
for ele in data.columns:
    if data[ele].dtype=='O':
        cat.append(ele)
        
print(cat)


# In[11]:


cat.remove('Item_Identifier')


# In[12]:


cat.remove('Outlet_Identifier')


# In[13]:


cat


# In[14]:


for ele in cat:
    print(ele)
    print(data[ele].value_counts())


# In[15]:


import seaborn as sns


# In[16]:


data['Item_Visibility'].value_counts()


# In[17]:


data['Item_Visibility'].replace([0],data['Item_Visibility'].median(),inplace=True)


# In[18]:


sum(data['Item_Visibility']==0)


# In[19]:


data['new_item_type']=data['Item_Identifier'].apply(lambda x:x[:2])


# In[20]:


data['new_item_type']


# In[21]:


data['new_item_type'] = data['new_item_type'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})


# In[22]:


data['new_item_type'].value_counts()


# In[23]:


data['Item_Fat_Content'].value_counts()


# In[24]:


data['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':"Low Fat",'reg':"Regular"},inplace=True)


# In[25]:


data['Item_Fat_Content'].value_counts()


# In[26]:


data.head()


# In[27]:


data['Outlet_years']=2022 - data['Outlet_Establishment_Year']


# In[28]:


data['Outlet_years']


# In[29]:


data.drop(data['Outlet_Establishment_Year'],inplace=True)


# In[30]:


data.head()


# In[31]:


sns.distplot(data['Item_Weight'])


# In[32]:


sns.distplot(data['Item_Visibility'])


# In[33]:


sns.distplot(data['Item_Outlet_Sales'])


# In[34]:


data['Item_Outlet_Sales'] = np.log(data['Item_Outlet_Sales'])


# In[35]:


sns.distplot(data['Item_Outlet_Sales'])


# In[36]:


sns.countplot(data['Item_Fat_Content'])


# In[37]:


sns.countplot(data['Item_Type'])


# In[38]:


sns.countplot(data['Outlet_years'])
plt.title('No of years of establishment')


# In[39]:


sns.countplot(data['Outlet_Size'])


# In[40]:


sns.countplot(data['Outlet_Location_Type'])


# In[41]:


sns.countplot(data['Outlet_Type'])


# In[42]:


data['Outlet_Type'].value_counts()


# # Correlation matrix

# In[43]:


sns.heatmap(data.corr(),annot=True)


# In[44]:


data.head()


# In[45]:


cat


# In[46]:


for ele in data.columns:
    print(ele)


# In[47]:


data.head()


# In[52]:


pd.get_dummies(data['Item_Fat_Content'],drop_first=True)


# In[53]:


data.drop('Outlet_Identifier',inplace=True,axis=1)


# In[49]:


num=[]
for ele in data.columns:
    if data[ele].dtype!='O':
        num.append(ele)
num


# In[55]:


data=pd.concat([data[num],
               pd.get_dummies(data['Item_Fat_Content'],drop_first=True),
               pd.get_dummies(data['Item_Type'],drop_first=True),
               pd.get_dummies(data['Outlet_Size'],drop_first=True),
               pd.get_dummies(data['Outlet_Type'],drop_first=True),
               pd.get_dummies(data['new_item_type'],drop_first=True)],axis=1)


# In[56]:


data.head()


# In[57]:


data['sales']=data['Item_Outlet_Sales']
data.head()


# In[58]:


data.drop('Item_Outlet_Sales',axis=1,inplace=True)


# In[60]:


for ele in data.columns:
    print(ele)


# In[59]:


data.head()


# In[61]:


data.drop('Outlet_Establishment_Year',axis=1,inplace=True)


# In[62]:


data.head()


# In[63]:


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[65]:


y


# In[66]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.3,random_state=0)


# In[67]:


#cross validation scored for different models
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics


# In[69]:


def train(model,x,y):
    model.fit(x,y)
    pred=model.predict(x)
    
    cv_score=cross_val_score(model,x,y,scoring='neg_mean_squared_error',cv=5)
    cv_score=np.abs(np.mean(cv_score))
    
    print('CV Score: ',cv_score)


# In[71]:


from sklearn.linear_model import LinearRegression,Ridge
model=LinearRegression(normalize=True)
train(model,x_train,y_train)


# In[72]:


model = Ridge(normalize=True)
train(model,x_train,y_train)


# In[73]:


from sklearn.linear_model import Lasso


# In[74]:


model=Lasso()
train(model,x_train,y_train)


# In[79]:


temp=[]
for ele in data.columns:
    if ele!='sales':
        temp.append(ele)


# In[83]:


p=pd.DataFrame(temp)


# In[88]:


from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
train(model,x_train,y_train)
coef=model.feature_importances_
print(coef)


# In[94]:


data.head()


# In[93]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
train(model,x_train,y_train)
coef=model.feature_importances_
print(coef)


# In[91]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
train(model,x_train,y_train)


# In[95]:


from xgboost import XGBRegressor
model = XGBRegressor()
train(model,x_train,y_train)


# # Hyperparameter tuning

# In[96]:


from sklearn.model_selection import RandomizedSearchCV


# # Random Forest Regressor

# In[97]:


max_features=['auto','sqrt']
max_depth =[int(x) for x in np.linspace(5,30,num=6)]
min_samples_split =[2,5,10,15,100]
min_samples_leaf=[1,2,5,10]


# In[98]:


random_grid={
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf
}
print(random_grid)


# In[99]:


rf = RandomForestRegressor()


# In[100]:


rf = RandomizedSearchCV(estimator =rf, param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=0)
rf.fit(x_train,y_train)


# In[101]:


print(rf.best_params_)


# In[102]:


print(rf.best_score_)


# In[103]:


pred = rf.predict(x_test)


# In[104]:


sns.distplot(y_test-pred)


# # LGBMRegressor

# In[106]:


from scipy.stats import uniform,randint
params={
    'gamma':uniform(0,0.5),
    'learning_rate':uniform(0.03,0.3),
    'max_depth':randint(2,6),
    'n_estimators':randint(100,150),
    'subsample':uniform(0.6,0.4)
    
}


# In[113]:


from lightgbm import LGBMRegressor


# In[114]:


lgb=LGBMRegressor()


# In[115]:


lgb=RandomizedSearchCV(estimator=model,param_distributions=params,scoring='neg_mean_squared_error',n_iter=10,cv=5,random_state=0)


# In[116]:


lgb.fit(x_train,y_train)


# In[117]:


print(lgb.best_params_)
print(lgb.best_score_)
red = lgb.predict(x_test)


# In[118]:


sns.distplot(y_test-pred)


# # XGBRegressor

# In[119]:


params={
    'gamma':uniform(0,0.5),
    'learning_rate':uniform(0.03,0.3),
    'max_depth':randint(2,6),
    'n_estimators':randint(100,150),
    'subsample':uniform(0.6,0.4)
    
}


# In[120]:


from xgboost import XGBRegressor


# In[121]:


xgb=XGBRegressor()


# In[122]:


xgb = RandomizedSearchCV(estimator=model,param_distributions=params,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=0)


# In[123]:


xgb.fit(x_train,y_train)


# In[124]:


print(xgb.best_params_)
print(xgb.best_score_)
pred = xgb.predict(x_test)


# In[125]:


sns.distplot(y_test-pred)


# In[ ]:




