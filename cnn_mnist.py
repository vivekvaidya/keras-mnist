# implementation of a simple 8 layered cnn trained on the mnist dataset.
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as kr

# parameters
classes = 10
batch_size = 128
iterations = 24

# split data between training and testing sets
(trainx, trainy), (testx, testy) = mnist.load_data()

imgx, imgy = 28, 28

if kr.image_data_format() == 'channels_first':
    trainx = trainx.reshape(trainx.shape[0], 1, imgx, imgy)
    testx = testx.reshape(testx.shape[0], 1, imgx, imgy)
    input_shape = (1, imgx, imgy)
else:
    trainx = trainx.reshape(trainx.shape[0], imgx, imgy, 1)
    testx = testx.reshape(testx.shape[0], imgx, imgy, 1)
    input_shape = (imgx, imgy, 1)

# preprocessing
trainx = trainx.astype('float32')
testx = testx.astype('float32')
trainx /= 255
testx /= 255

# setup class matrices
trainy = keras.utils.to_categorical(trainy, classes)
testy = keras.utils.to_categorical(testy, classes)

# setup the model, add layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

# compile model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# fit the model
model.fit(trainx, trainy, batch_size=batch_size, epochs=iterations, verbose=1, validation_data=(testx, testy))

# measure performance
score = model.evaluate(testx, testy, verbose=1)
print('\nTest accuracy: ', score[1])
