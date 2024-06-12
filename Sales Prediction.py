#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


Data=pd.read_csv("advertising.csv")


# In[5]:


Data


# In[6]:


Data.shape


# In[7]:


Data.isnull().sum()


# In[8]:


Data.describe()


# In[9]:


Data.info


# In[10]:


sns.distplot(Data.Sales) 


# In[11]:


sns.distplot(Data.Newspaper)


# In[12]:


sns.distplot(Data.TV)


# In[13]:


sns.distplot(Data.Radio)


# In[14]:


sns.pairplot(Data, x_vars=['TV', 'Newspaper', 'Radio'],y_vars='Sales',height=4,aspect=1, kind='scatter')
plt.show()


# In[15]:


plt.figure(figsize=(5,5))
sns.heatmap(Data.corr(), cmap="YlGnBu",annot=True,fmt='.2f')
plt.show()


# In[16]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[18]:


X = Data[['TV', 'Radio', 'Newspaper']]
Y = Data['Sales']


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[20]:


Model = LinearRegression()


# In[21]:


Model.fit(X_train, Y_train)


# In[22]:


X_train.head()


# In[23]:


Y_train.head()


# In[25]:


prediction= Model.predict(X_test)


# In[26]:


print(Model.coef_)
print(Model.intercept_)


# In[30]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, prediction)
print(f"Mean Squared Error: {mse}")


# In[31]:


Model.score(X_test, Y_test)


# In[37]:


import pandas as pd
new_data = pd.DataFrame([[100, 25, 10]])
new_sales_prediction = Model.predict(new_data)
print(f"Predicted Sales for New Data: {new_sales_prediction}")


# In[ ]:




