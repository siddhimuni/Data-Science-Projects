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


data = pd.read_csv('C:/Project datasets/Fraud.csv')


# In[3]:


data.head()


# In[4]:


y = data['isFraud']
y


# # EDA

# In[5]:


#finding all categorical variables
cat=[]
for col in data.columns:
    if data[col].dtypes=='O':
        cat.append(col)


# In[6]:


cat


# In[7]:


data.isnull().sum()


# In[8]:


#no missing values


# In[9]:


#for categorical variables
data['type'].unique()


# In[10]:


data['type'].nunique()


# In[11]:


#Applying one hot encode for type column
pd.get_dummies(data['type'], drop_first=True)


# In[12]:


data['nameOrig'].unique()


# In[13]:


data['nameOrig'].nunique()


# In[14]:


data['nameOrig'].value_counts()


# In[15]:


data['nameOrig'].value_counts().to_dict()


# In[15]:


data['nameOrig'].head()


# In[16]:


data_frequency_map = data['nameOrig'].value_counts().to_dict()


# In[17]:


data['nameOrig'] = data['nameOrig'].map(data_frequency_map)


# In[18]:


data['nameOrig'].head()


# In[19]:


data['nameOrig'].unique()


# In[20]:


#since nameOrig has very high cardinality, it was not possible to apply label encoding or one hot encoding otherwise it would result into a very high dimensionality dataset, we applied frequency/ count encoding which converts each value by the frequency of its occurence


# In[21]:


data['nameDest'].unique()


# In[22]:


data['nameDest'].nunique()


# In[23]:


data['nameDest'].value_counts()


# In[25]:


data['nameDest'].value_counts().to_dict()


# In[24]:


data_frequency_map = data['nameDest'].value_counts().to_dict()


# In[25]:


data['nameDest'] = data['nameDest'].map(data_frequency_map)


# In[26]:


data['nameDest'].head()


# In[27]:


data['nameDest'].unique()


# In[28]:


data['nameDest'].nunique()


# In[29]:


data.head()


# In[30]:


data['step'].unique()


# In[31]:


#for numerical variables
num=[]
for col in data.columns:
    if data[col].dtypes!='O':
        num.append(col)


# In[32]:


num


# In[33]:


num.remove('isFraud')


# In[34]:


num.remove('isFlaggedFraud')


# In[35]:


num


# In[36]:


data.head()


# In[37]:


#for statistical info
print(round(data[num].describe()),2)


# In[38]:


#amount has extreme outliers
#oldbalanceOrig has extreme outliers
#newbalanceOrig has extreme outliers
#nameDest has outliers
#olbalanceDest has outliers
#newbalanceDest has outliers


# In[39]:


plt.figure(figsize=(15,10))

plt.subplot(3,3,1)
fig=data.boxplot(column='amount')
fig.set_ylabel('amount')

plt.subplot(3,3,2)
fig=data.boxplot(column='oldbalanceOrg')
fig.set_ylabel('oldbalanceOrg')

plt.subplot(3,3,3)
fig=data.boxplot(column='newbalanceOrig')
fig.set_ylabel('newbalanceOrig')

plt.subplot(3,3,4)
fig=data.boxplot(column='nameDest')
fig.set_ylabel('nameDest')

plt.subplot(3,3,5)
fig=data.boxplot(column='oldbalanceDest')
fig.set_ylabel('oldbalanceDest')

plt.subplot(3,3,6)
fig=data.boxplot(column='newbalanceDest')
fig.set_ylabel('newbalanceDest')







# In[40]:


plt.figure(figsize=(15,10))

plt.subplot(3,3,1)
fig=data.amount.hist(bins=5)
fig.set_xlabel('amount')
fig.set_ylabel('isFraud')

plt.subplot(3,3,2)
fig=data.oldbalanceOrg.hist(bins=5)
fig.set_xlabel('oldbalanceOrg')
fig.set_ylabel('isFraud')

plt.subplot(3,3,3)
fig=data.newbalanceOrig.hist(bins=5)
fig.set_xlabel('newbalanceOrig')
fig.set_ylabel('isFraud')

plt.subplot(3,3,4)
fig=data.nameDest.hist(bins=5)
fig.set_xlabel('nameDest')
fig.set_ylabel('isFraud')

plt.subplot(3,3,5)
fig=data.oldbalanceDest.hist(bins=5)
fig.set_xlabel('oldbalanceDest')
fig.set_ylabel('isFraud')

plt.subplot(3,3,6)
fig=data.newbalanceDest.hist(bins=5)
fig.set_xlabel('newbalanceDest')
fig.set_ylabel('isFraud')


# In[41]:


#oldbalanceOrg , newbalanceOrig and nameDest are showing positiv/right skewness


# In[39]:


#finding outliers for oldbalanceOrg
IQR= data.oldbalanceOrg.quantile(0.75) - data.oldbalanceOrg.quantile(0.25)
Lower_fence = data.oldbalanceOrg.quantile(0.25)-(IQR*3)
Upper_fence = data.oldbalanceOrg.quantile(0.75)+(IQR*3)
print('oldbalanceOrg outliers are <{lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,upperboundary=Upper_fence))


# In[40]:


#finding outliers for newbalanceOrig
IQR= data.newbalanceOrig.quantile(0.75) - data.newbalanceOrig.quantile(0.25)
Lower_fence = data.newbalanceOrig.quantile(0.25)-(IQR*3)
Upper_fence = data.newbalanceOrig.quantile(0.75)+(IQR*3)
print('newbalanceOrig outliers are <{lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,upperboundary=Upper_fence))


# In[41]:


#finding outliers for nameDest
IQR= data.nameDest.quantile(0.75) - data.nameDest.quantile(0.25)
Lower_fence = data.nameDest.quantile(0.25)-(IQR*3)
Upper_fence = data.nameDest.quantile(0.75)+(IQR*3)
print('nameDest outliers are <{lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence,upperboundary=Upper_fence))


# In[42]:


print(round(data[num].describe()),2)


# In[43]:


y


# In[44]:


x = pd.concat([data[num],
              pd.get_dummies(data['type'], drop_first=True),
              ],axis=1)


# In[45]:


x


# In[46]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2,random_state=0)


# In[47]:


y_train.shape


# In[48]:


y_test.shape


# In[49]:


x_train.shape,x_test.shape


# In[50]:


#removing the outliers
def max_value(ele,variable,top):
    return np.where(ele[variable]>top,top,ele[variable])

for ele in [x_train,x_test]:
    ele['oldbalanceOrg']=max_value(ele,'oldbalanceOrg',429260.7)
    ele['newbalanceOrig']=max_value(ele,'newbalanceOrig',577033.64)
    ele['nameDest']=max_value(ele,'nameDest',65)


# In[51]:


print(round(x_train[num].describe()),2)


# In[52]:


#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# # Model training using various classifiers

# In[53]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=0)
logreg.fit(x_train,y_train)


# In[54]:


y_pred_test=logreg.predict(x_test)
y_pred_test


# In[55]:


y_pred_train=logreg.predict(x_train)
y_pred_train


# # Evaluation Matrix

# In[56]:


from sklearn.metrics import accuracy_score
print('Training set accuracy:{0:0.4f}'.format(accuracy_score(y_train,y_pred_train)))


# In[57]:


print('Test set accuracy:{0:0.4f}'.format(accuracy_score(y_test,y_pred_test)))


# # Confusion Matrix

# In[58]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_test)


# In[59]:


print("Confusion matrix\n",cm)
print('True positive= ',cm[0,0])
print('True negative= ',cm[1,1])
print('False positive= ',cm[0,1])
print('False negative= ',cm[1,0])


# # Classification report

# In[60]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_test))


# In[61]:


TP=cm[0,0]
TN=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]


# In[62]:


classification_accuracy=(TP+TN)/float(TP+TN+FP+FN)
print('classification accuracy: {0:0.4f}'.format(classification_accuracy))


# In[63]:


precision = TP/float(TP+FP)
print('precision: {0:0.4f}'.format(precision))


# In[64]:


recall = TP/float(TP+FN)
print('recall: {0:0.4f}'.format(recall))


# In[65]:


specificity = TN/float(TN+FP)
print('specificity: {0:0.4f}'.format(specificity))


# In[66]:


y_pred1=logreg.predict_proba(x_test)[:,1]


# In[67]:


from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# # Decision Tree Classifier

# In[68]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[69]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[70]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)


# In[72]:


print(classifier.predict(scaler.transform([[1,9839.64,1,170136.00,160296.36,1,0.00,0.00,0,0,1,0]])))


# In[73]:


y_pred_train=classifier.predict(x_train)
y_pred_train


# In[74]:


y_pred_test=classifier.predict(x_test)
y_pred_test


# In[75]:


print('Test score accuracy:{0:0.4f}'.format(accuracy_score(y_test,y_pred_test)))


# In[76]:


print('Train score accuracy:{0:0.4f}'.format(accuracy_score(y_train,y_pred_train)))


# In[77]:


dt_report = classification_report(y_test,y_pred_test)
print(dt_report)


# # Random Forest

# In[78]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, verbose=2)


# In[80]:


rf.fit(x_train,y_train)


# In[81]:


pred = rf.predict(x_test)


# In[82]:


print('Test score accuracy:{0:0.4f}'.format(accuracy_score(y_test,pred)))


# In[83]:


pred_train=rf.predict(x_train)


# In[84]:


print('Train score accuracy:{0:0.4f}'.format(accuracy_score(y_train,pred_train)))


# In[85]:


acc = rf.score(x_test,y_test)
print(acc)


# In[95]:


pred


# In[ ]:




