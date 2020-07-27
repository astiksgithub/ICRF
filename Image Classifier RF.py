
# coding: utf-8

# In[2]:


#importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#using pandas to read the database stored in the same folder
data = pd.read_csv("mnist.csv")


# In[4]:


#viewing column heads
data.head()


# In[10]:


#extracting data from the dataset and viewing then up close
a = data.iloc[4,1:].values


# In[11]:


#rshaping the extracted data into a resonable size
a = a.reshape(28,28).astype('uint8')


# In[12]:


plt.imshow(a)


# In[20]:


#preparing the data
#separating labels and data values
df_x = data.iloc[:,:-1]
df_y = data.iloc[:,-1]


# In[21]:


#creating test and train sizes/batches
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4)


# In[22]:


#check data
x_train.head()


# In[23]:


y_train.head()


# In[24]:


#call rf classifier
rf = RandomForestClassifier(n_estimators=100)


# In[25]:


#fit the model
rf.fit(x_train,y_train)


# In[26]:


#prediction on test data
pred = rf.predict(x_test)


# In[27]:


pred


# In[29]:


#check prediction accuracy
s = y_test.values

#calculate number of correctly predicted values
count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count += 1


# In[30]:


count


# In[32]:


#total values that the prediction code was run on
len(pred)


# In[34]:


#accuracy value
13558/14000

