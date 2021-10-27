import csv
import cv2
import numpy as np
'''
Obtencion del  path e imagenes
de la carpeta 'data'
'''

lines =[]
with open('data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

'''
Se aumenta la cantidad de imagenes
empleando la funcion flip de cv2
'''

augmented_images =[]
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D
from keras.layers import  MaxPooling2D

'''
Como se sugirio en los videos de Udacity
se empleo la red neuronal empleada en NVIDIA
'''

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape = (160, 320, 3)))
model.add(Lambda(lambda x : x/255.0 - 0.5))
model.add(Convolution2D(filters = 3, kernel_size =(5, 5), strides = (2,2) , activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(filters = 24, kernel_size =(5, 5), strides = (2,2) ,activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Convolution2D(filters = 36, kernel_size =(5, 5), strides = (2,2) ,activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Convolution2D(filters = 48, kernel_size =(3, 3), strides = (1,1)  ,activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Convolution2D(filters = 64, kernel_size =(3, 3), strides = (1,1) , activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, \
    epochs=7)
model.save('model_nvidia.h5')
exit()

