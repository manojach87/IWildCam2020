#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import readJson


#%%
root='C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\'
jsonTrainFilePath=root+'iwildcam2020_train_annotations.json'
jsonTestFilePath=root+'iwildcam2020_test_information.json'
trainPath=root+'train\\28X28\\'
testPath =root+'test\\100X100\\'
#%%
def getImageAsArray(path,df):
    from PIL import Image
    import os
    dataF=[{"file_name":file_name,
              "image":
                  np.concatenate(
                  np.asarray(Image.open(os.path.join(path, file_name)).convert('L')).tolist()
                  ).ravel().tolist()
              } 
             for file_name in df.file_name[:]]
    return dataF
#%%
def getTrainData(jsonFilePath):
    data = readJson.readJSONFile(jsonFilePath)

    annotations = data["annotations"]
    images=data["images"]
    categories = data["categories"]
    info = data["info"]

    # Convert to Data frame

    annotations = pd.DataFrame.from_dict(annotations)
    images = pd.DataFrame.from_dict(images)
    categories = pd.DataFrame.from_dict(categories)

    #Remove data from memory
    del data

    #Create column image_id to use for merging the two data frames
    images["image_id"]  = images["id"]

    # Merge annotations and images on image_id

    trainDf1 = (pd.merge(annotations, images, on='image_id'))
    #Remove Unnecessary fields
    trainDf1 = trainDf1.drop(["id_y","id_x"], axis = 1)

    #print(trainDf1.columns)
    
    # Unset annotations and images dataframe as they are no longer needed
    del annotations
    del images

    # Open the image from working directory

    trainDf=getImageAsArray(trainPath,trainDf1)

    #Convert to dataframe
    
    trainDf = pd.DataFrame.from_dict(trainDf)


    trainDf1=trainDf1.drop(['count', 'image_id', 'seq_id', 'width', 'height'],axis=1)

    # Merge the two training data frames to make sure the order is right and the image data is tied to file_name
    trainDf = (pd.merge(trainDf, trainDf1, on='file_name'))

    trainDf=trainDf.drop(['file_name', 'seq_num_frames', 'location','datetime', 'frame_num'],axis=1)

    # convert the dataframe to array so that it can be flattened, normal flatten() did not work
    trainDf=np.asarray(trainDf)

    # Concatenating the category_id and the image array of size=(100,100), this will flatten the data completely
    trainDf=[[arr[1]]+list(arr[0]) for arr in trainDf]
    
    # Converting back to dataframe
    trainDf=pd.DataFrame(trainDf)
    
    #Write the processed file to csv file for future use
    #trainDf.to_csv("trainDf.csv")
    
    return trainDf

#%%
def getTestData(jsonFilePath):
    data = readJson.readJSONFile(jsonTestFilePath)
    
    images=data["images"]
    categories = data["categories"]
    info = data["info"]
    
    # Convert to Data frame
    
    images = pd.DataFrame.from_dict(images)
    categories = pd.DataFrame.from_dict(categories)
    
    #Remove data from memory
    del data, info
    
    # Remove Unnecessary fields from images
    testDf1 = pd.DataFrame(images.file_name)
    
    # Unset images dataframe as it is no longer needed
    #del images
    
    # Open the image from working directory
    
    testDf=getImageAsArray(testPath,testDf1)
    
    #Convert to dataframe
    
    testDf = pd.DataFrame.from_dict(testDf)
    
    
    # Merge the two training data frames to make sure the order is right and the image data is tied to file_name
    testDf = (pd.merge(testDf, testDf1, on='file_name'))
    
    testDf=testDf.drop(['file_name'],axis=1)

    # convert the dataframe to array so that it can be flattened, normal flatten() did not work
    testDf=np.asarray(testDf)
    
    
    # Concatenating the category_id and the image array of size=(100,100), this will flatten the data completely
    testDf=[list(arr[0]) for arr in testDf]
    
    # Converting back to dataframe
    testDf=pd.DataFrame(testDf)
    
    #Write the processed file to csv file for future use
    testDf.to_csv("testDf.100X100.csv")
    
    return testDf
#%%
#trainDf=getTrainData(jsonTrainFilePath)
testDf=getTestData(jsonTestFilePath)

#from sklearn.ensemble import RandomForestClassifier

#%%
# Normalize the values
testDf  = testDf /255.0

#%%

testDf = testDf.values.reshape(-1,100,100,1)

#%%
import tensorflow as tf
model=tf.keras.models.load_model("model.1.save")

trainDf=pd.read_csv("trainDf.csv")
#%%
trainDf=trainDf.drop(columns=['Unnamed: 0'],axis=1)
#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainDf.drop(columns=["0"]), trainDf["0"], test_size=0.2)

#%%

X_train = X_train/255.0
X_test  = X_test /255.0


#%%
X_train = X_train.values.reshape(-1,100,100,1)
X_test = X_test.values.reshape(-1,100,100,1)
#%%

y_train = np.asarray(y_train)
X_train = np.asarray(X_train)


# In[6]:


model.fit(X_train, y_train, epochs=10)


#%%
preds = model.predict(testDf)
#%%
#%%
df = pd.DataFrame(data=preds, columns=range(0,267))


#%%


predList = df.idxmax(axis=1)