#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# In[46]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values


# In[47]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# In[48]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("dataset trained")


# In[49]:


viz_train = plt
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('hours vs scores (train)')
viz_train.show()

viz_test = plt
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('hours vs scores(test)')
viz_test.show()


# In[50]:


print(X_test)
y_pred = regressor.predict(X_test)


# In[51]:


df = pd.DataFrame({'actual': y_test, ' predicted' :y_pred})
df


# In[52]:


y_pred = regressor.predict(X_test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(y_pred[0]))


# In[53]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




