#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# In[2]:


data= pd.read_csv('IRIS.csv')


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.isnull().sum()


# In[7]:


print("Number of rows: ",data.shape[0])
print("Number of columns: ",data.shape[1])


# In[8]:


data.info()


# In[9]:


duplicate = data.duplicated().sum()
print(f'number of duplicated rows are {duplicate}')
     


# In[10]:


data.columns


# In[11]:


data.describe(include= 'all').round(2)


# In[12]:


for i in data.columns.tolist():
  print("No. of unique values in",i,"is",data[i].nunique())


# In[13]:


data1=data.iloc[:,1:]


# In[14]:


data1.head()


# In[15]:


sns.catplot(x = 'species', hue = 'species', kind = 'count', data = data1)


# In[16]:


plt.bar(data['species'],data['petal_width'])


# In[17]:


sns.set()
sns.pairplot(data[['sepal_length','sepal_width','petal_length','petal_width','species']], hue = "species", diag_kind="kde")


# In[18]:


data.describe()


# In[19]:


data.columns


# In[20]:


data.info()


# In[21]:


data


# In[22]:


ds= data.drop(['species'], axis=1)


# In[23]:


ds


# In[24]:


Label_Encode = LabelEncoder()
Y = data['species']
Y = Label_Encode.fit_transform(Y)


# In[25]:


Y


# In[26]:


data['species'].nunique()


# In[27]:


X = np.array(ds)


# In[28]:


X


# In[29]:


Y


# In[30]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)


# In[31]:


X_train


# In[32]:


X_train.shape


# In[33]:


X_test.shape


# In[34]:


Y_test.shape


# In[35]:


Y_train.shape


# In[36]:


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler().fit(X_train)
X_train_std = standard_scaler.transform(X_train)
X_test_std = standard_scaler.transform(X_test)


# In[37]:


X_train_std


# In[38]:


Y_train


# In[39]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_std,Y_train)


# In[40]:


prediction=knn.predict(X_test_std)
accuracy=accuracy_score(Y_test,prediction)*100


# In[41]:


accuracy


# In[42]:


X_train.size


# In[43]:


Y_train.size


# In[44]:


from sklearn import tree
Decision_tree = tree.DecisionTreeClassifier()
Decision_tree.fit(X_train,Y_train)


# In[46]:


prediction_tree=Decision_tree.predict(X_test)
accuracy_got=accuracy_score(Y_test,prediction_tree)*100


# In[47]:


accuracy_got


# In[ ]:




