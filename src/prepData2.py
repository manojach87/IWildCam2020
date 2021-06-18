#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
#import src.readJson as readJson

from PIL import Image
from datetime import datetime
import os, time

def readJSONFile(path):
    import json
    with open(path) as f:
        data = json.load(f)
    return data

# global categories
# global trainDf
# global trainDf1
# global annotations 

#%%
root='C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\'
jsonTrainFilePath=root+'iwildcam2020_train_annotations.json'
jsonTestFilePath=root+'iwildcam2020_test_information.json'
trainPath=root+'train\\28X28\\'
testPath =root+'test\\100X100\\'
#%%
global categories
global trainDf
global trainDf1
#data = readJson.readJSONFile(jsonFilePath)
jsonFilePath=jsonTrainFilePath
data = readJSONFile(jsonFilePath)

annotations = data["annotations"]
images=data["images"]
categories = data["categories"]
#info = data["info"]

# Convert to Data frame

annotations = pd.DataFrame.from_dict(annotations)
images = pd.DataFrame.from_dict(images)
categories = pd.DataFrame.from_dict(categories)

#Remove data from memory
del data

#categories["Sk"] = categories.index

#Create column image_id to use for merging the two data frames
images.rename(columns={"id": "image_id"},inplace = True)
#images["image_id"]  = images["id"]
annotations.drop(["id"], axis = 1, inplace=True)

# Merge annotations and images on image_id

categories.rename(columns={"id": "category_id"}, inplace=True)

trainDf1 = (pd.merge(annotations, images, on='image_id'))
trainDf1 = pd.merge(trainDf1,categories,on='category_id')

del annotations
del images

#%%

# categories["newCount"]=[    
#            row[2]=="human"              and int(row[0]/3)
#        or (row[2]=="empty"              and int(row[0]/8)
#        or (row[2]=="meleagris ocellata" and int(row[0]/2)
#        or row[0]
#        ))for row in np.asarray(categories)]
# categories.columns
#%%
x=trainDf1.groupby(['category_id']).count()["file_name"]
x=pd.DataFrame(x)

#%%

x["category_id"]=x.index
#%%
x.index=range(len(x))
#%%
x["Sk"]=x.index
#%%
categories=pd.merge(x,categories, on="category_id")
#%%
categories.rename(columns={"count":"old_count","file_name": "count"},inplace = True)
#%%

#%%
categories["sampleCount"]=[    
           row[4]=="human"              and int(row[0]/3)
       or (row[4]=="empty"              and int(row[0]/8)
       or (row[4]=="meleagris ocellata" and int(row[0]/2)
       or row[0]
       ))for row in np.asarray(categories)]
categories.columns
#%%
categories

#%%
mask = categories['Sk'].isin(trainDf1["Sk"])
result = categories[mask]

#%%
def sample(df,col,value,sample_size=100, random_state=100):
    if(sample_size>-1):
        return df[df[col]==value].sample(sample_size, random_state=random_state)
    else :
        return df[df[col]==value]

#%%
sample(trainDf1,"Sk",183,-1)

#%%
def createNewTrainingDf(dataframe):
    df=pd.DataFrame()
    for row in np.array(categories):
        col="category_id"
        value=row[1]
        sample_size=row[5]
        dfTemp=sample(dataframe,col,value,sample_size)
        dfTemp["Sk"]=row[2]
        df=df.append(dfTemp)       
    return df
#%%
trainDf1=createNewTrainingDf(trainDf1)
#%%

def getImageAsArray(path,df):
    dataF=[{"file_name":file_name,
              "image":
                  np.concatenate(
                  (np.asarray(Image.open(os.path.join(path, file_name)).convert('L'))) #.tolist()
                  ).ravel().tolist()
              } 
             for file_name in df.file_name[:]]
    return dataF
#%%
def getImageAsArray9(path,df):
    a=[]
    df1=pd.merge(df,categories.rename(columns={"id": "category_id"}),on='category_id')
    for row in np.array(df1[:]):
        a.append({
              "Sk": row[13],
              "image": np.concatenate((np.asarray(Image.open(os.path.join(path, row[10])).convert('L')))).tolist()
              }
        )
    return pd.DataFrame(a)#%%

#%%
    
def getTrainData(jsonFilePath):

    global trainDf
    #print(trainDf1.columns)

    # Unset annotations and images dataframe as they are no longer needed


    # Open the image from working directory
    
    start = time.time()
    
    #trainDf=getImageAsArray9(trainPath,trainDf1.sample(100))
    trainDf=getImageAsArray9(trainPath,trainDf1)

    end = time.time()
    print( str(end - start) + " seconds to load images from file.")
    
    #Convert to dataframe
    
    #trainDf = pd.DataFrame.from_dict(trainDf)
    global sk
    sk=trainDf["Sk"]
    start = time.time()
   # convert the dataframe to array so that it can be flattened, normal flatten() did not work
    #trainDf=np.asarray(trainDf)

    end = time.time()
    print( str(end - start) + " seconds to convert dataframe to array.")
    
    start = time.time()
    # Concatenating the category_id and the image array of size=(100,100), this will flatten the data completely
    #trainDf=[[arr[0]]+list(arr[1]) for arr in trainDf]
    
    trainDf=pd.DataFrame(trainDf['image'].values.tolist())#, columns=list(range(10000)))
    trainDf["Sk"]=sk
    end = time.time()
    print( str(end - start) + " seconds to flatten image array")
    
    start = time.time()
    # Converting back to dataframe
    #trainDf=pd.DataFrame(trainDf)
    #trainDf.rename(columns={0: "Sk"},inplace = True)
    
    end = time.time()
    print( str(end - start) + " seconds for other preprocessing.")
    #Write the processed file to csv file for future use
    #trainDf.to_csv("trainDf.csv")
    
    #trainDf.rename(columns={0: "file_name",1:"id"},inplace = True)
    
    #return categories, trainDf

#%%

def getTestData(jsonFilePath):
    data = readJSONFile(jsonTestFilePath)
    
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
    start = time.time()
    
    testDf=getImageAsArray(testPath,testDf1[:])
    
    end = time.time()
    print( str(end - start) + "seconds to read all the images")
    
    #Convert to dataframe
    
    testDf = pd.DataFrame.from_dict(testDf)
    
    
    # Merge the two training data frames to make sure the order is right and the image data is tied to file_name
    testDf = (pd.merge(testDf, testDf1, on='file_name'))
    
    testFiles=testDf["file_name"]
    testDf.drop(['file_name'],axis=1, inplace=True)

    # convert the dataframe to array so that it can be flattened, normal flatten() did not work
    testDf=np.asarray(testDf)

    # Concatenating the category_id and the image array of size=(100,100), this will flatten the data completely
    testDf=[list(arr[0]) for arr in testDf]

    # Converting back to dataframe
    testDf=pd.DataFrame(testDf)
    
    #Write the processed file to csv file for future use
    #testDf.to_csv("testDf.100X100.csv")
    
    return testFiles, testDf
# In[1]:
#categories, trainDf=getTrainData(jsonTrainFilePath)
#if 'trainDf' in locals() or 'trainDf' in globals():
#  del trainDf

getTrainData(jsonTrainFilePath)


#%%
#trainDf=pd.read_csv("trainDf.100000.csv")
#%%
trainDf.to_csv("train.144662.csv")
#trainDf=pd.read_csv("train.csv")
#.to_csv("train.csv")

# In[1]:
from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(trainDf.drop(columns=["0"]), trainDf["0"], test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(trainDf.drop(columns=["Sk"]), trainDf["Sk"], test_size=0.1)
#%%
#Skipping Test train Split
X_train=trainDf.drop(columns=["Sk"])
# X_train=trainDf.drop(columns=["Sk","Unnamed: 0"])
y_train=trainDf["Sk"]

#%%
#del trainDf


# In[2]:

X_train = X_train/255.0

#X_test  = X_test /255.0
# In[3]:


X_train.replace(0,1e-10, inplace=True)
#%%
#X_test =X_test.replace(0,1e-10)

#%%


#%%
# Replace very small number with zero  
#X_train[X_train == 1e-10] = 0

#%%
#X_train =X_train.replace(1e-10,0)
#%%
'''

X_train[X_train == 0] = 1e-10
X_test [X_test  == 0] = 1e-10

'''

# In[4]:
X_train = X_train.values.reshape(-1,100,100,1)
#%%
X_test = X_test.values.reshape(-1,100,100,1)
# In[5]:

y_train = np.asarray(y_train)
X_train = np.asarray(X_train)
# In[6]:

print(X_train[np.isnan(X_train)])
#print(X_test [np.isnan(X_test )])
print(y_train[np.isnan(y_train)])
#print(y_test [np.isnan(y_test )])

#%%
import tensorflow as tf
import tensorflow.keras as keras
#%%
#model=tf.keras.models.load_model("model.1.save")
#%%
model = keras.Sequential()
#%%
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (100,100,1)))
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(3,3)))
model.add(keras.layers.Dropout(0.25))


model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
model.add(keras.layers.Dropout(0.25))

# model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
# model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
# model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation = "relu"))
model.add(keras.layers.Dropout(0.5))
#model.add(keras.layers.Dense(267, activation = "softmax"))
model.add(keras.layers.Dense(len(categories), activation = "softmax"))

#model.add(keras.layers.Dense(len(categories), activation = "softmax", weights = [np.zeros([[100,100,1], len(categories)]), np.zeros(len(categories))]))

#optimizer = keras.optimizers.RMSprop(learning_rate=0.00, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
#optimizer = keras.optimizers.SGD()
#optimizer = keras.optimizers.RMSprop()

#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[6]:
model.load_weights(filepath='src/final_weight.h5')
#%%
model.load_weights(filepath='final_weight.conv2x2.1024.20200323.02.h5')
#%%
model.fit(X_train, y_train, epochs=15)
#%%
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test, batch_size=128)
print('test loss, test acc:', results)
#%%

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB'
)

#%%
model.save_weights(filepath='final_weight.conv2x2.1024.20200326.02.h5')
#%%
model.save('final_weight.conv2x2.1024.20200326.02.model.save')
#%%
from tensorflow.keras.models import load_model

model=load_model('final_weight.conv2x2.1024.20200326.02.model.save')
#%%
testFiles, testDf =getTestData(jsonTestFilePath)

#%%
testDf.to_csv("testDf.csv")
#%%
pd.DataFrame(testFiles).to_csv("testFiles.csv")
# In[2]:
# Normalize the values
testDf = testDf /255.0

#%%

#X_train.replace(0,1e-10, inplace=True)

#%%

testDf = testDf.values.reshape(-1,100,100,1)

#%%
# Replace very small number with zero  
testDf[testDf == 0] = 1e-10

#%%
preds = model.predict(testDf)
##%%
##%%
preds = pd.DataFrame(data=preds, columns=range(0,len(categories)))


##%%


preds = preds.idxmax(axis=1)

##%%
preds=pd.DataFrame(preds)
##%%

preds['index1'] = preds.index
##%%
preds["file_name"]=testFiles
##%%

#predList["Sk"]=predList[0]
preds.rename(columns={0: "Sk"},inplace = True)
##%%
#predList=predList.drop(columns=[0],axis=1)
##%%

preds = (pd.merge(preds, categories, on="Sk"))

##%%
#.drop(columns=["",""])
##%%
#del X_train
#%%
##%%
#predList["Category"]=predList["id"]
preds.rename(columns={"category_id": "Category"},inplace = True)

##%%
preds.drop(columns=["Sk","count","name","old_count", "sampleCount"], axis=1, inplace = True)
##%%
preds.sort_values("index1", axis = 0, ascending = True, inplace = True, na_position ='last')
##%%
#predList=predList.drop(columns=["0_x","0_y"], axis=1)
##%%
##predList = (pd.merge(predList, , on="Sk"))
#preds["file_name"]=testFiles
#%%
data = readJSONFile(jsonTestFilePath)
    
images=data["images"]

images = pd.DataFrame.from_dict(images)
##%%
images = images[["id","file_name"]]
# categories = pd.DataFrame.from_dict(categories)
#%%
preds=(pd.merge(preds, images, on="file_name"))[["id","Category"]]

#%%
submission=pd.read_csv("../data/iwildcam-2020/sample_submission.csv").drop(columns=["Category"],axis=1)
##%%
preds.rename(columns={"id": "Id"},inplace = True)

##%%
preds=(pd.merge(submission, preds, on="Id")) #[["Id","Category"]]

#%%
now = datetime.now()
current_time = now.strftime("%Y%m%d%H%M%S")

#%%
preds.to_csv("submissison."+current_time+".csv",index=False)

#%%


