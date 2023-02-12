#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train=pd.read_csv('C:/TDS Data Science/train.csv')


# In[3]:


train


# In[4]:


y=train['Survived']
y


# In[5]:


#for col in train.columns:
 #   print(col)


# 
# 
# # EDA

# # Finding all the categorical data

# In[6]:


categorical = ['Name','Sex','Ticket','Cabin','Embarked','Survived']


# In[7]:


train[categorical].isnull().sum()


# # Exploring all the categorical variables individually

# In[8]:





train['Name'].unique()


# In[9]:


train['Name'].nunique()


# In[10]:


train['Name'].value_counts()


# In[11]:


pd.get_dummies(train['Name'],drop_first=True)


# In[12]:


train['Sex'].unique()


# In[13]:


train['Sex'].value_counts()


# In[14]:


pd.get_dummies(train['Sex'],drop_first='True').head()


# In[15]:


train['Ticket'].unique()


# In[16]:


pd.get_dummies(train['Ticket'],drop_first=True)


# In[17]:


train['Cabin'].unique()


# In[18]:


train['Cabin'].nunique()


# In[19]:


train['Cabin'].value_counts()


# In[20]:


pd.get_dummies(train['Cabin'],drop_first=True,dummy_na=True)


# In[21]:


train['Embarked'].unique()


# In[22]:


train['Embarked'].nunique()


# In[23]:


pd.get_dummies(train['Embarked'],drop_first=True,dummy_na=True)


# In[24]:


train.head()


# # Exploring Numerical data

# In[25]:


numerical = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']


# In[26]:


train[numerical].isnull().sum()
#age has major missing values


# In[27]:


print(round(train[numerical].describe()),2)
#Fare has extreme outliers
#Age ,SibSp and Parch has partial outliers


# In[28]:


plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig=train.boxplot(column='Fare')
fig.set_ylabel('Fare')

plt.subplot(2,2,2)
fig=train.boxplot(column='Age')
fig.set_ylabel('Age')

plt.subplot(2,2,3)
fig=train.boxplot(column='SibSp')
fig.set_ylabel('SibSp')

plt.subplot(2,2,4)
fig=train.boxplot(column='Parch')
fig.set_ylabel('Parch')


# In[29]:


plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
fig=train.Fare.hist(bins=5)
fig.set_xlabel('Fare')
fig.set_ylabel('Survived')

plt.subplot(2,2,2)
fig=train.Age.hist(bins=5)
fig.set_xlabel('Age')
fig.set_ylabel('Survived')

plt.subplot(2,2,3)
fig=train.SibSp.hist(bins=5)
fig.set_xlabel('SibSp')
fig.set_ylabel('Survived')

plt.subplot(2,2,4)
fig=train.Parch.hist(bins=5)
fig.set_xlabel('Parch')
fig.set_ylabel('Survived')


# # For numerical data, Fare,SibSp and Parch shows positive skewness
# 
# 

# In[30]:


#find outliers for Fare varibale
IQR= train.Fare.quantile(0.75) - train.Fare.quantile(0.25)
Lower_fence = train.Fare.quantile(0.25)-(IQR*3)
Upper_fence = train.Fare.quantile(0.75)+(IQR*3)
print('Fare outliers are <{lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,upperboundary=Upper_fence))


# In[31]:


#find outliers for SibSp varibale

IQR= train.SibSp.quantile(0.75) - train.SibSp.quantile(0.25)
Lower_fence = train.SibSp.quantile(0.25)-(IQR*3)
Upper_fence = train.SibSp.quantile(0.75)+(IQR*3)
print('SibSp outliers are <{lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,upperboundary=Upper_fence))


# In[32]:


#find outliers for Parch varibale
IQR= train.Parch.quantile(0.75) - train.Parch.quantile(0.25)
Lower_fence = train.Parch.quantile(0.25)-(IQR*3)
Upper_fence = train.Parch.quantile(0.75)+(IQR*3)
print('Parch outliers are <{lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,upperboundary=Upper_fence))


# In[33]:


print(round(train[numerical].describe()),2)


# In[34]:


train


# In[35]:


train['Cabin'].isnull().sum()


# In[36]:


train['Age'].isnull().sum()


# In[37]:


train['Age'].fillna(train['Age'].median(),inplace=True)


# In[38]:


train['Age'].isnull().sum()


# In[39]:


train['Cabin'].fillna(train['Cabin'].mode()[0],inplace=True)


# In[40]:


train['Cabin'].isnull().sum()


# In[41]:


train.isnull().sum()


# In[42]:


x=train=pd.concat([train[numerical],
                 pd.get_dummies(train['Name']),
                 pd.get_dummies(train['Sex']),
                 pd.get_dummies(train['Ticket']),
                 pd.get_dummies(train['Cabin']),
                 pd.get_dummies(train['Embarked'])],axis=1)


# In[43]:


from sklearn.model_selection import train_test_split
x_train,x_test=train_test_split(x,test_size=0.2,random_state=0)


# In[44]:


x_train.shape,x_test.shape


# In[45]:


y


# In[46]:


y_train,y_test=train_test_split(y,test_size=0.2,random_state=0)


# In[47]:


y_train.shape,y_test.shape


# # Split the data into training and test set

# In[48]:


train[numerical].isnull().sum()


# In[49]:


x_train.isnull().sum()


# # Engineering outliers in numerical variables

# In[50]:


def max_value(ele,variable,top):
    return np.where(ele[variable]>top,top,ele[variable])

for ele in [x_train,x_test]:
    ele['Fare']=max_value(ele,'Fare',100)
    ele['SibSp']=max_value(ele,'SibSp',4)
    ele['Parch']=max_value(ele,'Parch',0)


# In[51]:


x_train[numerical].describe()
#the outliers have been removed


# In[52]:


x_train


# # Feature Scaling

# In[53]:


x_train.describe()


# In[54]:


cols=x_train.columns


# In[55]:


x_train.shape


# In[56]:


x_test.shape


# In[57]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[58]:


y_train = y_train.replace(np.nan,' ',regex=True)
y_test = y_test.replace(np.nan,' ',regex=True)


# # Model Training

# In[59]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=0)
logreg.fit(x_train,y_train)


# In[60]:


y_pred_test=logreg.predict(x_test)
y_pred_test


# In[61]:


y_pred_train=logreg.predict(x_train)
y_pred_train


# # Evaluation Matrix

# In[62]:


from sklearn.metrics import accuracy_score
print('Training set accuracy:{0:0.4f}'.format(accuracy_score(y_train,y_pred_train)))


# In[63]:


print('Test set accuracy:{0:0.4f}'.format(accuracy_score(y_test,y_pred_test)))


# # Confusion Matrix

# In[64]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_test)


# In[65]:


print("Confusion matrix\n",cm)
print('True positive= ',cm[0,0])
print('True negative= ',cm[1,1])
print('False positive= ',cm[0,1])
print('False negative= ',cm[1,0])


# In[67]:


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# # Classification Report

# In[68]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_test))


# In[69]:


from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred_train))


# In[70]:


TP=cm[0,0]
TN=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]


# In[71]:


classification_accuracy=(TP+TN)/float(TP+TN+FP+FN)
print('classification accuracy: {0:0.4f}'.format(classification_accuracy))


# In[72]:


classification_error=(FP+FN)/float(TP+TN+FP+FN)
print('classification accuracy: {0:0.4f}'.format(classification_error))


# # Precision

# In[73]:


precision = TP/float(TP+FP)
print('precision: {0:0.4f}'.format(precision))


# # Recall

# In[74]:


recall = TP/float(TP+FN)
print('recall: {0:0.4f}'.format(recall))


# # Specificity

# In[75]:


specificity = TN/float(TN+FP)
print('specificity: {0:0.4f}'.format(specificity))


# In[83]:


y_pred1=logreg.predict_proba(x_test)[:,1]


# In[85]:


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred1, pos_label = 'Yes')

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for RainTomorrow classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()


# In[88]:


from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# In[90]:


from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, x_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


# In[93]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, x_train, y_train, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))
print('Average cross-validation score: {:.4f}'.format(scores.mean()))


# In[ ]:




