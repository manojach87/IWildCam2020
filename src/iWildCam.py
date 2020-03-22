#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import readJson


#%%
jsonFilePath='C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\iwildcam2020_train_annotations.json'

data = readJson.readJSONFile(jsonFilePath)

#%%
# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
#print(data)
#print(data.keys())
annotations = data["annotations"]
images=data["images"]
categories = data["categories"]
info = data["info"]

#%%
# Convert to Data frame

annotations = pd.DataFrame.from_dict(annotations)
images = pd.DataFrame.from_dict(images)
categories = pd.DataFrame.from_dict(categories)

#%%
categories["category_id"]=categories["id"]
categories["Sk"]=[i for i in range(len(categories))]

#%%
#Remove data from memory
del data

#%%
#Create column image_id to use for merging the two data frames
images["image_id"]  = images["id"]

#%%
#annotations = (pd.merge(annotations, categories, on='category_id'))
#%%
# Merge annotations and images on image_id

trainDf1 = (pd.merge(annotations, images, on='image_id'))
#Remove Unnecessary fields
#%%
trainDf1 = trainDf1.drop(["id_y","id_x"], axis = 1)

print(trainDf1.columns)
#%%
# Unset annotations and images dataframe as they are no longer needed
del annotations
del images
#%%
from PIL import Image
# Open the image form working directory
trainPath='C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\train\\28X28\\'

#print(trainDf.file_name[:10])

def getImage(filePath,h=28,w=28):
    image=Image.open(filePath).convert('LA')
    image.thumbnail((h,w))
    #image.show()
    return image
    #images["filename"]=np.asarray(image)

#print(getImage(trainPath+'8c74033c-21bc-11ea-a13a-137349068a90.jpg'))

#images=[{"file_name":file_name,"image":np.asarray(getImage(trainPath+file_name))} for file_name in trainDf.file_name[:]]

# for file_name in trainDf.file_name[:10] :
#     image=Image.open(trainPath+file_name).convert('LA')
#     image.thumbnail(200,200)
#%%
trainDf=[{"file_name":file_name,
          "image":
              np.concatenate(
              np.asarray(Image.open(trainPath+file_name).convert('L')).tolist()
              ).ravel().tolist()
          } 
         for file_name in trainDf1.file_name[:]]
#images=[[file_name,Image.open(trainPath+file_name).convert('LA')] for file_name in trainDf.file_name[:10]]

#trainDf["image"]= Image.open(trainPath+trainDf.file_name[1]).convert('LA')
#%%
trainDf = pd.DataFrame.from_dict(trainDf)

#%%
#print(np.asarray(trainDf.image.tolist()).reshape(100,1,100,100))
#images["image"]= np.asarray(images["image"])
#%%
#print(len(trainDf.file_name))

#%%
# print(type(images["image"]  ))
# print(images["image"].shape)
# x=[np.asarray(images["image"][i]) for i in range(len(images["image"]))]
# print(x)

#images=[np.asarray(Image.open(trainPath+file_name).convert('LA')) for file_name in trainDf.file_name[:]][10000]


# images["x"]=x
#del images
#%%

trainDf1=trainDf1.drop(['count', 'image_id', 'seq_id', 'width', 'height'],axis=1)
#%%
trainDf = (pd.merge(trainDf, trainDf1, on='file_name'))



#%%
#x=trainDf[:10]
#print(x.columns)

trainDf=trainDf.drop(['file_name', 'seq_num_frames', 'location','datetime', 'frame_num'],axis=1)


#%%
#print(trainDf.image[0].tolist())
#%%
#x=trainDf
#%%
#trainDf=x
#%%
#trainDf["image"]=[trainDf.image[i].tolist()[0] for i in range(len(trainDf.image))]
#trainDf["image"]=[trainDf.image[i].reshape(100,100,2) for i in range(len(trainDf.image))]
#%%
#pd.DataFrame(trainDf).to_csv("trainDf.csv")


#%%
trainDf=np.asarray(trainDf)
#%%
trainDf=[[arr[1]]+list(arr[0]) for arr in trainDf]
#%%
#trainDf=np.asarray(trainDf[1])
#%%
trainDf=pd.DataFrame(trainDf)
#%%
trainDf.to_csv("trainDf.csv")

#%%
#%%
#%%trainDf["img"]=[list(arr[0])+list(arr[1]) for arr in trainDf.image]

#%%
#x["image"]=x[0]
#print(len(np.array(np.array(x.image).flatten())))
#print(len(np.array(np.array(x.image).flatten()).tolist()))

#%%
#print((trainDf.image)[1].shape)
#print((trainDf.image.values))
#print(trainDf.drop(columns=0))

# In[3]:


#from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(trainDf.drop(columns=["category_id"]), trainDf.category_id, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(trainDf.drop(columns=0), trainDf[0], test_size=0.2)
#X_train, X_test, y_train, y_test = train_test_split(trainDf["image"], trainDf["category_id"], test_size=0.2)

#%%

X_train = X_train/255.0
X_test  = X_test /255.0



#%%


# X_train = X_train.reshape(X_train.shape[0], 100, 100, 2)
# X_test = X_test.reshape(X_test.shape[0], 100, 100, 2)
# input_shape = (100, 100, 1)

#%%
X_train = X_train.values.reshape(-1,100,100,1)
X_test = X_test.values.reshape(-1,100,100,1)

#%%

#%%
#import numpy as np

#X_train = np.arange(len(X_train)).reshape(len(X_train),2, 10000 )

#X_train = X_train.reshape((X_train.shape[0]*2, 28, 28))

#print(X_train.shape)

# In[4]:


# model = keras.Sequential([
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
#     ])

model = keras.Sequential()

model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (100,100,1)))
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))


model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.25))


model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1000, activation = "relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(267, activation = "softmax"))

#optimizer = keras.optimizers.RMSprop(learning_rate=0.00, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
#optimizer = keras.optimizers.SGD()
#optimizer = keras.optimizers.RMSprop()

#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[5]:


y_train = np.asarray(y_train)
X_train = np.asarray(X_train)


# In[6]:

#%%
import pandas as pd
y_train_1he = pd.get_dummies(y_train)

#%%

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(y_train.reshape(len(y_train),1,))   
#enc.n_values_
enc.n_values_

#%%
model.fit(X_train, y_train_1he, epochs=10)

#%%
model.fit(X_train, y_train, epochs=240)

#%%
model.save("model.1.save")
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


del trainDf, X_train, X_test

