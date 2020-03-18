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
#%%
with open('C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\iwildcam2020_train_annotations.json') as f:
  data = json.load(f)
#%%
# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
#print(data)
print(data.keys())
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
images["image_id"]  = images["id"]
#%%
tainDf= None
trainDf = (pd.merge(annotations, images, on='image_id'))
trainDf = trainDf.drop(["id_y","id_x"], axis = 1)

print(trainDf.columns)
#%%
# Images and Seq ID
image_id = "901c0b74-21bc-11ea-a13a-137349068a90"
seq_id = str(images[images["id"]==image_id]["seq_id"])
print(seq_id)
print(annotations[annotations["image_id"]==image_id])
print(trainDf[trainDf["seq_id"]==seq_id])
#%%
# read an image and print on screen
picLocation = 'C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\iwildcam-2020\\train\\400X400\\'+image_id+'.jpg'

def showImage(loc):
    import matplotlib as plt
    from matplotlib import pyplot 
    im = pyplot.imread(loc)
    pyplot.imshow(im)

showImage(picLocation)