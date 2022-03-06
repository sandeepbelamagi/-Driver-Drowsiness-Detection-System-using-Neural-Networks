"""
Drowsiness detection of humans
"""
import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

new_model = tf.keras.models.load_model('my_model.h5')

# print(new_model.summary())

img = cv2.imread('pexels-photo-614810.jpeg')

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in eyes:
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyess = eyeCascade.detectMultiScale(roi_gray)
    if len(eyess) == 0:
        print("eyes not detected")
    else:
        for ex, ey, ew, eh in eyess:
            eyes_roi = roi_color[ey:ey + eh, ex:ex + ew]

plt.imshow(cv2.cvtColor(eyes_roi, cv2.COLOR_BGR2RGB))

plt.show()

final_img = cv2.resize(eyes_roi, (224,224))
final_img = np.expand_dims(final_img, axis=0)
final_img = final_img/255.0

print(new_model.predict(final_img))
