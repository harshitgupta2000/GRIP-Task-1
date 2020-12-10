#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation

# # GRIP 

# # Name : Harshit Gupta

# # Task-1 Prediction Using Supervised ML

# Predict the percentage of students based on the number of study hours

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[2]:


#Reading The Data
url="http://bit.ly/w-data"
data = pd.read_csv(url)


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.tail()


# In[7]:


data.describe()


# In[9]:


data.shape


# In[10]:


data.info


# In[12]:


#Plotting the distribution graphs
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Study Hours')
plt.ylabel('Percentage Score')
plt.show()


# In[13]:


x=data.iloc[:, :-1].values
y=data.iloc[:, 1].values


# In[14]:


x


# In[15]:


y


# In[16]:


x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,random_state=0)


# # Training the Model

# In[18]:


from sklearn.linear_model import LinearRegression
re = LinearRegression()
re.fit(x_train,y_train)
print("Trained")


# In[20]:


#Regression Graph
line = re.coef_*x + re.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# # Prediction 

# In[21]:


y_pred = re.predict(x_test)
print(y_test)


# In[22]:


print(y_pred)


# In[23]:


#prediction for 9.25 hours/day
prediction = re.predict([[9.25]])
print("Number of hours: 9.25")
print("Prediction : {}".format(prediction[0]))


# # Evaluation

# In[24]:


from sklearn import metrics
err = metrics.mean_absolute_error(y_test, y_pred)
print("mean absolute error : {}".format(err))


# In[ ]:




