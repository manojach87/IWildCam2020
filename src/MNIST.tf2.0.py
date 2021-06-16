# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:17:48 2020

@author: manoj
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#%%

print(f"Tensor Flow Version: {tf.__version__}")
#print(f"Keras Version: {tensorflow.keras.__version__}")
#print()
#print(f"Python {sys.version}")
#print(f"Pandas {pd.__version__}")
#print(f"Scikit-Learn {sk.__version__}")
#%%
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")



#%%
data = keras.datasets.fashion_mnist

#%%

(train_images, train_labels),(test_images, test_labels) = data.load_data()


#%%
print(train_labels[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#%%
train_images = train_images/255.0
test_images = test_images/255.0
#print(train_images[7])
#%%

# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

#%%
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#%%

model.fit(train_images, train_labels, epochs=10)

#%%
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested "+ str(test_acc))

#%%
prediction= model.predict(test_images)
#%%
print(np.argmax(prediction[0]))

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual    : "+ class_names[test_labels[i] ])
    plt.title ("Predicted : "+ class_names[np.argmax(prediction[i])])
    plt.show()
#print(class_names[np.argmax(prediction[0])])



