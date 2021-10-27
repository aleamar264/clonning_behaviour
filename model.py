import csv
import cv2
import numpy as np
import os
from math import ceil

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from random import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            augmented_images =[]
            augmented_measurements = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                left_image = cv2.imread('data/IMG/'+batch_sample[1].split('/')[-1])
                right_image = cv2.imread('data/IMG/'+batch_sample[2].split('/')[-1])
                center_angle = float(batch_sample[3])
                left_angle = center_angle +  correction
                right_angle = center_angle - correction

                images.extend((center_image, left_image, right_image))
                angles.extend((center_angle, left_angle, right_angle) )
                # images.append(np.fliplr(np.asarray(images)))
                # angles.append(-np.asarray(angles))
                # augmented_images.append(cv2.flip(images,1))
                # augmented_measurements.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D
from keras.layers import  MaxPooling2D, Dropout



model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape = (160, 320, 3)))
model.add(Lambda(lambda x : x/255.0 - 0.5))
model.add(Convolution2D(filters = 20, kernel_size =(5, 5), strides = (2,2) , activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(filters = 40, kernel_size =(5, 5), strides = (2,2) ,activation='relu', padding ="same"))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Convolution2D(filters = 80, kernel_size =(5, 5), strides = (2,2) ,activation='relu', padding ="same"))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Convolution2D(filters = 120, kernel_size =(3, 3), strides = (1,1)  ,activation='relu', padding ="same"))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Convolution2D(filters = 140, kernel_size =(3, 3), strides = (1,1) , activation='relu', padding ="same"))
model.add(Flatten())
model.add(Dense(120, activation = "relu"))
model.add(Dense(84, activation = "relu"))
model.add(Dense(43, activation = "relu"))
# model.add(Dropout(0.75))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch =ceil(len(train_samples*3)/batch_size), \
                    validation_data = validation_generator, \
                    validation_steps = ceil(len(validation_samples*3)/batch_size),
                    epochs = 5, verbose = 1)
model.save('model.h5')
model.summary()
exit()


