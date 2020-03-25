#%%
import pandas as pd
import numpy as np

def readJSONFile(path):
    import json
    with open(path) as f:
        data = json.load(f)
    return data

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:54:30 2020

@author: manoj
"""


def getTrain1Data(jsonFilePath):
    global categories
    global trainDf
    global trainDf1
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

    #Create column image_id to use for merging the two data frames
    images["image_id"]  = images["id"]

    # Merge annotations and images on image_id

    trainDf1 = (pd.merge(annotations, images, on='image_id'))
    #Remove Unnecessary fields
    trainDf1.drop(["id_y","id_x"], axis = 1, inplace=True)

    # Unset annotations and images dataframe as they are no longer needed
    del annotations
    del images

    trainDf1.drop(['count', 'image_id', 'seq_id', 'width', 'height', 'seq_num_frames', 'location','datetime', 'frame_num'],axis=1, inplace=True)

#%%

root='C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\'
jsonTrainFilePath=root+'iwildcam2020_train_annotations.json'
jsonTestFilePath=root+'iwildcam2020_test_information.json'
trainPath=root+'train\\28X28\\'
testPath =root+'test\\100X100\\'

getTrain1Data(jsonTrainFilePath)

#%%
#
print([(cnt/10 if name < "empty" else cnt) if for cnt, name in categories[["count","name"]])
[row[0]/3 if row[2]=="human" else row[0] for row in np.asarray(categories)]
#%%
'''
       # Program to demonstrate conditional operator 
a, b = 10, 20
  
# Copy value of a in min if a < b else copy b 
min = a/10 if a < b else b 
  
print(min) 
'''