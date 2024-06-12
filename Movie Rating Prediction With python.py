#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('Movie-rating.csv', encoding='latin1')


# In[4]:


data.head(10)


# In[5]:


data.tail(10)


# In[6]:


data.shape


# In[7]:


print('Number_of_Rows',data.shape[0])
print('Number_of_Columns',data.shape[1])


# In[8]:


data.info()


# In[9]:


data.isnull().sum()


# In[10]:


sns.heatmap(data.isnull())
plt.show()


# In[11]:


data = data.dropna(axis=0)


# In[12]:


sns.heatmap(data.isnull())
plt.show()


# In[13]:


data.isnull().sum()


# In[15]:


duplicate_data=data.duplicated().any()
print("Any duplicate data is present or not?",duplicate_data)


# In[17]:


data.columns


# In[19]:


data['Genre'].value_counts()


# In[20]:


data.describe()


# In[24]:


data.nunique()


# In[26]:


data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')


# In[27]:


sns.barplot(x='Year',y='Votes',data=data)
plt.title("Votes By Year")
plt.show()


# In[34]:


data.groupby('Director')['Rating'].mean().sort_values(ascending=False)


# In[36]:


sns.countplot(x='Year',data=data)
plt.title("Number of Movies Per Year")


# In[37]:


data.columns


# In[40]:


top_10 = data.nlargest(10, 'Rating')[['Rating', 'Director']].reset_index()


# In[41]:


top_10


# In[45]:


sns.barplot(x=top_10.index, y='Rating', data=top_10)
plt.title("Display Top 10 Highest Rated Movie Titles")


# In[50]:


data1 = data.groupby('Year')[['Rating']].mean().sort_values(by='Rating', ascending=False).reset_index()


# In[51]:


data1


# In[53]:


sns.barplot(x=data1.index, y='Rating', data=data1)


# In[54]:


def rating(rating):
    if rating>=7.0:
        return 'Excellent'
    elif rating>=6.0:
        return 'Good'
    else:
        return 'Average'


# In[57]:


data['rating_cat']=data['Rating'].apply(rating)
print(rating(8.5))  
print(rating(6.5))  
print(rating(4.5)) 


# In[58]:


from sklearn.preprocessing import LabelEncoder


# In[64]:


cat_cols=['Rating','Director','Genre','Actor 1','Actor 2','Actor 3']
le=LabelEncoder()  
for i in cat_cols:
    data[i]=le.fit_transform(data[i])
data.dtypes


# In[67]:


import seaborn as sns
import matplotlib.pyplot as plt

numerical_cols = data.select_dtypes(include=['int', 'float']).columns
rows = len(numerical_cols) // 2
cols = 2
fig, ax = plt.subplots(rows, cols, figsize=(15, 10))

index = 0
for i in range(rows):
    for j in range(cols):
        if index < len(numerical_cols):
            sns.distplot(data[numerical_cols[index]], ax=ax[i][j])
            ax[i][j].set_title(numerical_cols[index])
            index += 1

plt.tight_layout()
plt.show()


# In[68]:


data.columns


# In[69]:


skewed_features=['Name', 'Year', 'Duration',
       'Genre', 'Rating', 'Votes',
       'Director', 'Actor 1', 'Actor 2', 'Actor 3',
       'rating_cat']


# In[71]:


import numpy as np
for i in skewed_features:
    if np.issubdtype(data[i].dtype, np.number):  # Check if column is numeric
        data[i] = np.log(data[i] + 1)


# In[74]:


import seaborn as sns
import matplotlib.pyplot as plt
for i, col in enumerate(skewed_features):
    if np.issubdtype(data[col].dtype, np.number):  # Check if column is numeric
        plt.figure(figsize=(10, 5))
        sns.distplot(data[col])
        plt.title(f"Distribution of {col}")
        plt.show()


# In[77]:


X=data.drop(labels=['Rating'],axis=1)
Y=data['Rating']
X.head()


# In[78]:


Y.head()


# In[80]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[85]:


from sklearn.tree import DecisionTreeRegressor

non_numeric_columns = X.select_dtypes(exclude=['number']).columns

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for column in non_numeric_columns:
    X[column] = label_encoder.fit_transform(X[column])


remaining_non_numeric_columns = X.select_dtypes(exclude=['number']).columns
if len(remaining_non_numeric_columns) == 0:
    print("All features are numeric. Ready to train the model.")
else:
    print("Some features are still non-numeric. Further processing needed.")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

DT = DecisionTreeRegressor(max_depth=9)
DT.fit(X_train, Y_train)

train_preds = DT.predict(X_train)
test_preds = DT.predict(X_test)

from sklearn import metrics
RMSE_train = np.sqrt(metrics.mean_squared_error(Y_train, train_preds))
RMSE_test = np.sqrt(metrics.mean_squared_error(Y_test, test_preds))
print("RMSE Training Data:", RMSE_train)
print("RMSE Test Data:", RMSE_test)

print('R-squared value on train:', DT.score(X_train, Y_train))
print('R-squared value on test:', DT.score(X_test, Y_test))


# In[86]:


errors = abs(test_preds - Y_test)


# In[87]:


mape = 100 * (errors / Y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

