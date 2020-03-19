# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:59:36 2020

@author: manoj
"""


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


import os

path="C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\digit-recognizer"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
submissionPath=path+"\\sample_submission.csv"
sampleSubmissionDf= pd.read_csv(path+"\\sample_submission.csv")
#sampleSubmissionDf.head()
testDf= pd.read_csv(path+"\\test.csv")
#testDf.head()
trainDfX= pd.read_csv(path+"\\train.csv")
#print(trainDf.head())

trainDfX.index +=1
#trainDf.index

#trainDf.drop(columns=["label"]) #, axis=1


# In[3]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainDfX.drop(columns=["label"]), trainDfX.label, test_size=0.2)

X_train = X_train/255.0
X_test  = X_test /255.0

X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)


# In[4]:


model = keras.Sequential([
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model = keras.Sequential()

model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))


model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.25))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation = "relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation = "softmax"))

#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[5]:


y_train = np.asarray(y_train)
X_train = np.asarray(X_train)


# In[6]:


model.fit(X_train, y_train, epochs=10)


# In[11]:


#testDf = np.asarray(testDf)
testDf = testDf/255.0
testDf = testDf.values.reshape(-1,28,28,1)
testDf = np.asarray(testDf)


# In[ ]:


preds = model.predict(testDf)
preds


# In[ ]:


X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
test_loss, test_acc = model.evaluate(X_test, y_test)


# In[18]:


type(preds)


# In[14]:


train_x = trainDf.drop(columns=["label"])/255.0
train_y = trainDf.label

train_x = train_x.values.reshape(-1,28,28,1)



train_x = np.asarray(train_x)
train_y = np.asarray(train_y)


# In[15]:


model.fit(train_x, train_y, epochs=10)


# In[48]:

preds = model.predict(testDf)

#%%
df = pd.DataFrame(data=preds, columns=range(0,10))


# In[49]:


predList = df.idxmax(axis=1)


# In[50]:


submissionDf=pd.DataFrame({'Label': predList[:, ]})
submissionDf.index +=1

submissionDf["ImageId"] = submissionDf.index
submissionDf=submissionDf[["ImageId","Label"]]


# In[51]:


submissionDf


# In[52]:


submissionDf.to_csv(path+"\\sample_submission.csv",sep=",",index=False)


# In[ ]:




