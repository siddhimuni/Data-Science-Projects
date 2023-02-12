#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


data = pd.read_csv('C:/Loan Prediction Dataset.csv')
data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


sns.distplot(data['LoanAmount'])


# In[8]:


data['LoanAmount'].fillna(data['LoanAmount'].median(),inplace=True)


# In[9]:


data.isnull().sum()


# In[10]:


cat=[]
for ele in data.columns:
    if data[ele].dtype=='O':
        cat.append(ele)
cat


# In[11]:


data['Credit_History'].unique()


# In[12]:


sns.distplot(data['Credit_History'])


# In[13]:


data['Credit_History'].fillna(data['Credit_History'].median(),inplace=True)


# In[14]:


data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median(),inplace=True)


# In[15]:


data.isnull().sum()


# In[16]:


data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace=True)


# In[17]:


data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)


# In[18]:


data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)


# In[19]:


data.isnull().sum()


# In[20]:


data['Married'].fillna(data['Married'].mode()[0]
                       
                       ,inplace=True)


# In[21]:


data.isnull().sum()


# In[22]:


data.head()


# In[23]:


sns.countplot(data['Gender'])


# In[24]:


sns.countplot(data['Married'])


# In[25]:


sns.countplot(data['Dependents'])


# In[26]:


sns.countplot(data['Education'])


# In[27]:


sns.countplot(data['Self_Employed'])


# In[28]:


sns.countplot(data['Property_Area'])


# In[29]:


sns.countplot(data['Loan_Status'])


# In[30]:


sns.distplot(data['ApplicantIncome'])


# In[31]:


sns.distplot(data['CoapplicantIncome'])


# In[32]:


sns.distplot(data['LoanAmount'])


# In[33]:


sns.distplot(data['Loan_Amount_Term'])


# In[34]:


sns.distplot(data['Credit_History'])


# In[35]:


data.head()


# In[36]:


data['Total_income']=data['ApplicantIncome']+data['CoapplicantIncome']


# In[37]:


data.head()


# Applying log transformations for numerical data

# In[38]:


data['ApplicantIncome'] = np.log(data['ApplicantIncome'])
sns.distplot(data['ApplicantIncome'])


# In[40]:


data['LoanAmount'] = np.log(data['LoanAmount'])
sns.distplot(data['LoanAmount'])


# In[41]:


data['Loan_Amount_Term'] = np.log(data['Loan_Amount_Term'])
sns.distplot(data['Loan_Amount_Term'])


# In[42]:


data['Total_income'] = np.log(data['Total_income'])
sns.distplot(data['Total_income'])


# # Correlation Matrix

# In[43]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(),annot=True)


# In[44]:


data.head()


# In[45]:


cols=['Loan_ID','CoapplicantIncome']
data=data.drop(columns=cols,axis=1)


# In[46]:


data.head()


# In[47]:


cat


# In[48]:


cat.remove('Loan_ID')


# In[49]:


cat


# In[50]:


pd.get_dummies(data['Gender'],drop_first=True)


# In[51]:


pd.get_dummies(data['Married'],drop_first=True)


# In[53]:


pd.get_dummies(data['Dependents'],drop_first=True)


# In[54]:


pd.get_dummies(data['Education'],drop_first=True)


# In[55]:


pd.get_dummies(data['Self_Employed'],drop_first=True)


# In[56]:


pd.get_dummies(data['Property_Area'],drop_first=True)


# In[57]:


pd.get_dummies(data['Loan_Status'],drop_first=True)


# In[59]:


num=[]
for ele in data.columns:
    if data[ele].dtype!='O':
        num.append(ele)


# In[60]:


num


# In[61]:


data = pd.concat([data[num],
                 pd.get_dummies(data['Gender'],drop_first=True),
                 pd.get_dummies(data['Married'],drop_first=True),
                 pd.get_dummies(data['Dependents'],drop_first=True),
                 pd.get_dummies(data['Education'],drop_first=True),
                 pd.get_dummies(data['Self_Employed'],drop_first=True),
                 pd.get_dummies(data['Property_Area'],drop_first=True),
                 pd.get_dummies(data['Loan_Status'],drop_first=True)],axis=1)


# In[62]:


data.head()


# In[63]:


x=data.drop('Y',axis=1)


# In[64]:


x


# In[65]:


y=data['Y']


# In[66]:


y


# In[67]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3,random_state=0)


# # Model Training

# In[68]:


from sklearn.model_selection import cross_val_score


# In[69]:


def classify(model,x,y):
    x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3,random_state=0)
    model.fit(x_train,y_train)
    print('Accuracy:',model.score(x_test,y_test)*100)
    score=cross_val_score(model,x,y,cv=5)
    print('Cross validation score:',np.mean(score)*100)
    


# In[70]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
classify(model,x,y)


# In[71]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
classify(model,x,y)


# In[73]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
classify(model,x,y)


# In[74]:


from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
classify(model,x,y)


# In[75]:


from xgboost import XGBClassifier


# In[84]:


xgb=XGBClassifier()


# In[81]:


from lightgbm import LGBMClassifier


# In[82]:


lg=LGBMClassifier()


# In[87]:


get_ipython().system('pip install catboost')


# In[88]:


from catboost import CatBoostClassifier


# In[89]:


ctb=CatBoostClassifier()


# # Hyperparameter tuning

# In[91]:


from sklearn.model_selection import RandomizedSearchCV


# In[101]:


n_estimators =[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(5,30,num=6)]
min_samples_split=[2,5,10,15,100]
min_samples_leaf = [1,2,5,10]


# In[102]:


random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf
}


# In[103]:


print(random_grid)


# In[104]:


rf=RandomForestClassifier()


# In[105]:


rf=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='accuracy',n_iter=10,cv=5,verbose=2,n_jobs=1,random_state=42)


# In[106]:


rf.fit(x,y)


# In[107]:


rf.best_score_


# In[108]:


rf.best_params_


# In[109]:


max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(5,30,num=6)]
min_samples_split=[2,5,10,15,100]
min_samples_leaf = [1,2,5,10]


# In[110]:


param={
    
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf
}


# In[111]:


print(param)


# In[112]:


dt=DecisionTreeClassifier()


# In[113]:


dt=RandomizedSearchCV(estimator=dt,param_distributions=param,scoring='accuracy',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[114]:


dt.fit(x,y)


# In[115]:


dt.best_score_


# In[116]:


dt.best_params_


# In[117]:


ex=ExtraTreesClassifier()


# In[118]:


ex=RandomizedSearchCV(estimator=ex,param_distributions=random_grid,scoring='accuracy',n_iter=10,cv=5,verbose=2,n_jobs=1,random_state=42)


# In[119]:


ex.fit(x,y)


# In[120]:


ex.best_score_


# In[121]:


ex.best_params_


# In[122]:


from scipy.stats import uniform,randint


# In[123]:


xgb=XGBClassifier()


# In[135]:


params={
    'gamma':uniform(0,0.5),
    'learning_rate':uniform(0.03,0.3),
    'max_depth':randint(2,6),
    'n_estimators':randint(100,150),
    'subsample':uniform(0.6,0.4)
}


# In[136]:


xgb=RandomizedSearchCV(estimator=xgb,param_distributions=params,scoring='accuracy',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[134]:


xgb.fit(x,y)


# In[ ]:




