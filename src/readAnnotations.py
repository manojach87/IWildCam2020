# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:25:34 2020

@author: manoj
"""

#%%
import numpy as np
import pandas as pd
import cv2
import json
import readJson  # in the file called readJson.py

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
#Remove data from memory
data=None

#%%
#Create column image_id to use for merging the two data frames
images["image_id"]  = images["id"]

#%%
# Merge annotations and images on image_id

trainDf = (pd.merge(annotations, images, on='image_id'))
#Remove Unnecessary fields
trainDf = trainDf.drop(["id_y","id_x"], axis = 1)

print(trainDf.columns)
#%%
# Unset annotations and images dataframe as they are no longer needed
annotations = images = None


#%%
# Images and Seq ID
image_id = "901c0b74-21bc-11ea-a13a-137349068a90"

# read an image and print on screen
picLocation = 'C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\train\\400X400\\'+image_id+'.jpg'

import readImage as ri # Import readImage.py 

ri.showImage(picLocation)

