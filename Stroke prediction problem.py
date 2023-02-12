#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv('C:/train_2v.csv')
test = pd.read_csv('C:/test_2v.csv')


# In[3]:


train.head()


# In[4]:


categorical=[]
for ele in train.columns:
    if train[ele].dtype=='O':
        categorical.append(ele)


# In[5]:


print(categorical)


# In[6]:


train.isnull().sum()


# In[7]:


sns.distplot(train['bmi'])


# In[8]:


train['smoking_status'].unique()


# In[9]:


for col in train.columns:
    if col=='bmi':
        col_median = train[col].median()
        train[col].fillna(col_median,inplace=True)


# In[10]:


train.isnull().sum()


# In[11]:


test.isnull().sum()


# In[12]:


for col in test.columns:
    if col=='bmi':
        col_median = test[col].median()
        test[col].fillna(col_median,inplace=True)


# In[13]:


test.isnull().sum()


# In[14]:



        
        train['smoking_status'].fillna(train['smoking_status'].mode()[0],inplace=True)


# In[15]:


test['smoking_status'].fillna(test['smoking_status'].mode()[0],inplace=True)


# In[16]:


test.isnull().sum()


# In[17]:


train.isnull().sum()


# In[18]:


train.shape


# In[19]:


test.shape


# In[20]:


train.head()


# In[21]:


train['stroke'].value_counts()


# In[22]:


sns.countplot(x=train['gender'],hue=train['stroke'])
plt.title('gender vs stroke')
plt.show()


# In[23]:


sns.heatmap(train.corr(),annot=True)


# In[24]:


train['smoking_status'].value_counts()


# In[25]:


train.groupby(['gender'])['smoking_status'].value_counts()


# In[26]:


sns.countplot(x=train['gender'],hue=train['smoking_status'])
plt.title('gender vs type of smokers')
plt.show()


# In[27]:


train.head()


# In[28]:


categorical = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status','stroke']


# In[29]:


numerical=['id','age','avg_glucose_level','bmi']


# In[30]:


train['work_type'].unique()


# In[31]:


pd.get_dummies(train['gender'],drop_first=True).head()


# In[32]:


pd.get_dummies(train['hypertension'],drop_first=True).head()


# In[33]:


pd.get_dummies(train['heart_disease'],drop_first=True).head()


# In[34]:


pd.get_dummies(train['ever_married'],drop_first=True).head()


# In[35]:


pd.get_dummies(train['work_type'],drop_first=True).head()


# In[36]:


pd.get_dummies(train['Residence_type'],drop_first=True).head()


# In[37]:


pd.get_dummies(train['smoking_status'],drop_first=True).head()


# In[38]:


train.head()


# In[39]:


train.isnull().sum()


# In[40]:


y_train=train['stroke']


# In[41]:


x_train = pd.concat([train[numerical],
                   pd.get_dummies(train['gender'],drop_first=True),
                   pd.get_dummies(train['hypertension'],drop_first=True),
                   pd.get_dummies(train['heart_disease'],drop_first=True),
                   pd.get_dummies(train['ever_married'],drop_first=True),
                   pd.get_dummies(train['work_type'],drop_first=True) ,
                   pd.get_dummies(train['Residence_type'],drop_first=True),
                   pd.get_dummies(train['smoking_status'],drop_first=True)],
                   axis=1)
                


# In[42]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x_train,y_train)


# In[43]:


x_train


# In[44]:


x_test


# In[45]:


y_train


# In[46]:


y_test


# In[47]:


x_train.shape


# In[48]:


x_train.columns


# In[49]:


x_test.shape


# In[50]:


y_test.shape


# In[51]:


y_train.shape


# In[52]:


x_train.head()


# # Classification using Naive Bayes

# In[53]:


x_test.head()


# In[54]:


from sklearn.naive_bayes import GaussianNB


# In[55]:


model = GaussianNB()
model.fit(x_train,y_train)


# In[56]:


pred = model.predict(x_test)
pred


# In[57]:


test_score = model.score(x_test,y_test)
print(test_score)


# In[58]:


nb_conf_mtr = pd.crosstab(y_test,pred)


# In[59]:


nb_conf_mtr


# In[61]:


from sklearn.metrics import classification_report


# In[62]:


nb_report = classification_report(y_test,pred)


# In[64]:


print(nb_report)


# # Decision Trees

# In[65]:


from sklearn.tree import DecisionTreeClassifier


# In[81]:


dt = DecisionTreeClassifier(max_depth=10)
dt.fit(x_train,y_train)


# In[75]:


pred = dt.predict(x_test)
pred


# In[76]:


dt_score = dt.score(x_test,y_test)


# In[77]:


dt_score


# # Reports for decision tree

# In[78]:


dt_report = classification_report(y_test,pred)
print(dt_report)


# In[79]:


dt_conf_mtr = pd.crosstab(y_test,pred)
dt_conf_mtr


# # Random Forest

# In[83]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, verbose=2)


# In[84]:


rf.fit(x_train,y_train)


# In[85]:


pred = rf.predict(x_test)


# In[87]:


rf_conf = pd.crosstab(y_test,pred)
print(rf_conf)


# In[89]:


print(classification_report(y_test,pred))


# In[90]:


acc = rf.score(x_test,y_test)
print(acc)


# # Using Neural Networks

# In[91]:


from sklearn.neural_network import MLPClassifier


# In[92]:


mlp=MLPClassifier()


# In[93]:


mlp.fit(x_train,y_train)


# In[94]:


pred = mlp.predict(x_test)


# In[95]:


mlp.score(x_test,y_test)


# # Cross validating the accuracies

# In[97]:


#For naive bayes
from sklearn.model_selection import cross_val_score
print(cross_val_score(model,x_train,y_train,cv=20,scoring='accuracy').mean())


# # Applying PCA

# In[98]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)


# In[99]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x_train,y_train)


# In[100]:


model_2=GaussianNB()
model_2.fit(x_train,y_train)


# In[101]:


test_score=model_2.score(x_test,y_test)
print(test_score)


# In[104]:


dt_mod = DecisionTreeClassifier(max_depth=8)
dt_mod.fit(x_train,y_train)


# In[105]:


dt_score=dt_mod.score(x_test,y_test)
print(dt_score)


# In[107]:


rf.fit(x_train,y_train)


# In[108]:


rf.score(x_test,y_test)


# In[109]:


mlp=MLPClassifier()
mlp.fit(x_train,y_train)


# In[110]:


mlp.score(x_test,y_test)


# In[ ]:




