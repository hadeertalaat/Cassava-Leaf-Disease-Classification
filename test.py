import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import sys
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import LabelBinarizer

# image_size = 224
image_size = 100

model = keras.models.load_model('../input/funny-leaf-5/saved_model.h5')

def preprocess_test(path):
    arrays_t = []
    labels = []
    i = 0
    for dirname, _, filenames in (os.walk(path)):
        for filename in filenames:
#             if i >= 5:
            if i >= len(filenames):
                break
            i = i + 1
            image_path = os.path.join(path, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_size, image_size))
            arrays_t.append(image)  
            labels.append(filename)  
    arrays_t = np.array(arrays_t, dtype="float32")
    return arrays_t,labels
path = '/kaggle/input/cassava-leaf-disease-classification/test_images'
test_data,test_labels = preprocess_test(path)
test_data = test_data / 255.

y_perd = model.predict(test_data)
classes = [np.argmax(element) for element in y_perd]
output = pd.DataFrame({'image_id': test_labels, 'label': classes})
print(output)
output.to_csv('submission.csv', index=False)
