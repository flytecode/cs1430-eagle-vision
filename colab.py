import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import tensorflow_hub as hub
import matplotlib.pyplot as plt

train_dir = (r'C:\Users\Andy Zhou\cs1430\cs1430-eagle-vision\cs1430-eagle-vision\data\test')
test_dir = (r'C:\Users\Andy Zhou\cs1430\cs1430-eagle-vision\cs1430-eagle-vision\data\train')

data_args = dict(rescale=1./255, validation_split=.20, rotation_range=10, shear_range=5,
                height_shift_range=0.1, width_shift_range=0.1, horizontal_flip=True,
                brightness_range=[0.75, 1.25])

bag_train = tf.keras.preprocessing.image.ImageDataGenerator(**data_args)

train_gen = bag_train.flow_from_directory(
train_dir,
subset="training",
shuffle=True,
target_size=(64, 64))


bag_val = tf.keras.preprocessing.image.ImageDataGenerator(**data_args)

val_gen = bag_val.flow_from_directory(
train_dir,
subset="validation",
shuffle=True,
target_size=(64, 64))


for image_batch, label_batch in train_gen:
  break
  image_batch.shape, label_batch.shape

print (train_gen.class_indices)

base_model = tf.keras.Sequential([
 Conv2D(16, 3, 1, padding='same',
               activation='relu', name='block1_conv1'),
               Conv2D(32, 3, 1, padding='same',
               activation='relu', name='block1_conv2'),
               MaxPool2D(3, name='block1_pool'),

               # Layer 2
               Conv2D(32, 3, 1, padding='same',
               activation='relu', name='block2_conv1'),
               Conv2D(64, 3, 1, padding='same',
               activation='relu', name='block2_conv2'),
               MaxPool2D(3, name='block2_pool'),

               # Layer 3
               Conv2D(128, 5, 1, padding='same',
               activation='relu', name='block3_conv1'),
               Conv2D(256, 5, 1, padding='same',
               activation='relu', name='block3_conv2'),
               MaxPool2D(3, name='block3_pool'),
               
               BatchNormalization(),

               Flatten(),
               Dropout(0.4),

               # Output Layer
               Dense(units=200, activation='softmax')
])
base_model.build([None, 64, 64, 3])
base_model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

base_model.compile(
 optimizer=optimizer,
 loss= 'categorical_crossentropy',
 metrics=['accuracy'])

epochs=1
history = base_model.fit(
  train_gen,
  validation_data = val_gen,
  epochs = epochs,
  shuffle=True
)

print(history.history.keys())

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

import pickle
from sklearn.externals import joblib 

file_dir = 'cs1430-eagle-vision\cs1430-eagle-vision\Code\colab_model.h5'
base_model.save(file_dir)