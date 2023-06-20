#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 


# In[10]:


df=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')


# In[11]:


df


# In[12]:


df.shape


# In[13]:


df.describe()


# In[14]:


df.info()


# In[15]:


df.dtypes


# In[16]:


sns.scatterplot('Hours','Scores',data=df)


# In[17]:


x=df.iloc[:,:1]
y=df.iloc[:,1:]


# In[18]:


x


# In[19]:


model=LinearRegression()


# In[20]:


model.fit(x,y)


# In[21]:


model.score(x,y)


# ## What will be predict score if a student studies for 9.25 hrs/day?

# In[22]:


model.predict([[9.25]])


# In[ ]:




