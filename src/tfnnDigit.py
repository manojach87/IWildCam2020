# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

path="C:\\Users\\manoj\\PycharmProjects\\tf-tuto\\data\\digit-recognizer"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

sampleSubmissionDf= pd.read_csv(path+"\\sample_submission.csv")
#sampleSubmissionDf.head()
testDf= pd.read_csv(path+"\\test.csv")
#testDf.head()
trainDf= pd.read_csv(path+"\\train.csv")
#print(trainDf.head())

trainDf.index +=1
#trainDf.index

trainDf.drop(columns=["label"]) #, axis=1

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainDf.drop(columns=["label"]), trainDf.label, test_size=0.2)

model = keras.Sequential([
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10)


preds = model.predict(testDf)

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Tested "+ str(test_acc))

#%%
prediction= model.predict(test_images)
#%%
print(np.argmax(prediction[0]))


pd.DataFrame(data=preds[:,],    # values
index=preds[:,],    # 1st column as index
columns=["Label"])  # 1st row as the column names

submissionDf=pd.DataFrame({'Label': preds[:, ]})
submissionDf.index +=1

submissionDf["ImageId"] = submissionDf.index
submissionDf=submissionDf[["ImageId","Label"]]

submissionDf.to_csv("",sep=",",index=False)

keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="Same", activation="relu")

