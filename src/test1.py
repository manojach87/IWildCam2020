#%%
import pandas as pd
import numpy as np

def readJSONFile(path):
    import json
    with open(path) as f:
        data = json.load(f)
    return data



global categories
global trainDf
global trainDf1
global annotations 

#%%
root='C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\'
jsonTrainFilePath=root+'iwildcam2020_train_annotations.json'
jsonTestFilePath=root+'iwildcam2020_test_information.json'
trainPath=root+'train\\28X28\\'
testPath =root+'test\\100X100\\'
#%%
#def getTrain1Data(jsonFilePath):
jsonFilePath=jsonTrainFilePath
data = readJSONFile(jsonFilePath)


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

categories["Sk"] = categories.index


#Create column image_id to use for merging the two data frames
images["image_id"]  = images["id"]

# Merge annotations and images on image_id

trainDf1 = (pd.merge(annotations, images, on='image_id'))
#Remove Unnecessary fields
trainDf1.drop(["id_y","id_x"], axis = 1, inplace=True)

# Unset annotations and images dataframe as they are no longer needed
#del annotations
del images

#trainDf1.drop(['count', 'image_id', 'seq_id', 'width', 'height', 'seq_num_frames', 'location','datetime', 'frame_num'],axis=1, inplace=True)

#%%
#getTrain1Data(jsonTrainFilePath)

#%%
def getImageAsArray(path,df):
    dataF=[{"file_name":file_name,
              "image":
                  np.concatenate(
                  (np.asarray(Image.open(os.path.join(path, file_name)).convert('L'))) #.tolist()
                  ).tolist()
              } 
             for file_name in df.file_name[:]]
    return dataF
#%%
from PIL import Image
import os
def getImageAsArray1(path,df):
    dataF=[[file_name]
           + np.concatenate(
               np.asarray(
                   Image.open(os.path.join(path, file_name))
                   .convert('L')
               )  #/255.0
             ).tolist()
             for file_name in df.file_name[:]]
    #dataF=pd.DataFrame(dataF)
    return dataF
#%%
def getImageAsArray2(path,df):
    dataF=[[file_name]
           + [np.concatenate(
               np.asarray(
                   Image.open(os.path.join(path, file_name))
                   .convert('L')
               )  #/255.0
             )] 
           for file_name in df.file_name[:]]
    return dataF
#%%
def getImageAsArray3(path,df):
    dataF=[[file_name]
           + [np.asarray(
                   Image.open(os.path.join(path, file_name))
                   .convert('L')
               )]
             for file_name in df.file_name[:]]
    return dataF
#%%
def getImageAsArray4(path,df):
    dataF=[[file_name]
           + [Image.open(os.path.join(path, file_name))
                   .convert('L')
              ]
             for file_name in df.file_name[:]]
    return dataF
#%%
def getImageAsArray5(path,df):
    dataF=[[file_name]
           + [Image.open(os.path.join(path, file_name)).convert('L')]
             for file_name in df.file_name[:]]
    return dataF
#%%
def getImageAsArray6(path,df):
    dataF=[{"file_name":file_name,
              "image":
                  Image.open(os.path.join(path, file_name)).convert('L') #.tolist()
              } 
             for file_name in df.file_name[:]]
    return dataF
#%%
def getImageAsArray7(path,df):
    dataF = list()
    for file_name in df.file_name[:]:
        dataF.append({"file_name":file_name,"image": np.concatenate(
                  (np.asarray(Image.open(os.path.join(path, file_name)).convert('L'))) #.tolist()
                  ).tolist()
                      })
    return dataF
#%%
def getImageAsArray8(path,df):
    dataF=[{"file_name":row["file_name"][0],
              "image":
                  np.concatenate(
                  (np.asarray(Image.open(os.path.join(path, row["file_name"][0])).convert('L'))) #.tolist()
                  ).tolist()
              } 
             for row in df[:] ]
    return dataF
#%%
def getImageAsArray8(path,df):
    dataF=[row for row in df[:] ]
    return dataF
#%%
def double(x):
    return 2*x

dbl = lambda x: (2*x)

sk = lambda id: np.asarray(categories[categories["id"]==id])[0][3]

# x=trainDf1[:10]

# y=pd.DataFrame(list(map(sk,(x["category_id"]))))
# x["Sk"]=y[0]

#trainDf2=trainDf1

#trainDf2["Sk"]=list(map(sk,(trainDf2["category_id"])))

#%%
print(categories[categories["id"]==2]["Sk"])

#%%
import time
#%%

#del trainDf10
start = time.time()
trainDf10=getImageAsArray(trainPath,trainDf1[:10000])
end = time.time()
print(end - start)
del trainDf10
#%%
start = time.time()
trainDf10=getImageAsArray1(trainPath,trainDf1[:10000])
end = time.time()
print(end - start)
del trainDf10

start = time.time()
trainDf10=getImageAsArray2(trainPath,trainDf1[:10000])
end = time.time()
print(end - start)
del trainDf10

start = time.time()
trainDf10=getImageAsArray3(trainPath,trainDf1[:10000])
end = time.time()
print(end - start)
del trainDf10

start = time.time()
trainDf10=getImageAsArray4(trainPath,trainDf1[:10000])
end = time.time()
print(end - start)
del trainDf10

start = time.time()
trainDf10=getImageAsArray5(trainPath,trainDf1[:10000])
end = time.time()
print(end - start)
del trainDf10
#%%
start = time.time()
trainDf10=getImageAsArray9(trainPath,trainDf1.sample(100000))
end = time.time()
t1=end - start
print(t1)
#%%

start = time.time()
trainDf10=pd.DataFrame(trainDf10)
#trainDf10["image"]=flatten(list(map(np.array,trainDf10["image"])))
end = time.time()
t2=end - start
print(t1+t2)
#%%
path=trainPath

a=pd.merge(trainDf1[:10],categories.rename(columns={"id": "category_id"}),on='category_id')

def getImageAsArray8(path,df):
    a=pd.DataFrame()
    df1=pd.merge(df,categories.rename(columns={"id": "category_id"}),on='category_id')
    for row in np.array(df1[:]):
        if (row[0]<=row[11]):
            a=a.append({
                  #"file_name":row[10],
                  #"select": row[0]<=row[11],
                  "Sk": row[13],
                  "image": np.concatenate((np.asarray(Image.open(os.path.join(path, row[10])).convert('L')))).tolist()
                  #row[[10,2,3]] #+
                      }, ignore_index="True" ##, inplace=True
            )
    return a
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
    return pd.DataFrame(a)

#%%
flatten = lambda l : [item for sublist in l for item in sublist]

#%%


#%%

categories["newCount"]=[    
           row[2]=="human"              and int(row[0]/3)
       or (row[2]=="empty"              and int(row[0]/8)
       or (row[2]=="meleagris ocellata" and int(row[0]/2)
       or row[0]
       ))for row in np.asarray(categories)]
#%%

#%%
categoriesFromTrainData=categories["id"].unique()
#%%
categoriesFromTrainData=pd.DataFrame(categoriesFromTrainData)
categoriesFromTrainData.rename(columns={0: "id"},inplace = True)
categoriesFromTrainData.sort_values("id", axis = 0, ascending = True, 
              inplace = True, na_position ='last')

categoriesFromTrainData["Sk"]=[i for i in range(0,len(categoriesFromTrainData))]

categories1 = (pd.merge(categoriesFromTrainData, categories, on='id'))
#%%

print(len(trainDf[trainDf["Sk"]==36]))

print(trainDf[trainDf["Sk"]==36].sample(n=2000, random_state=1))

#%%
trainDf2=pd.DataFrame()
for row in np.asarray(categories1):
    traiDf2=trainDf2.append(trainDf[trainDf["Sk"]==row[1]].sample(n=row[4], random_state=1))

