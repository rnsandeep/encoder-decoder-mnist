from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.utils import to_categorical
import tensorflow as tf
import pickle

import numpy as np


train_inputs, train_labels = pickle.load(open('encoder_train.pkl','rb'))

train_labels = to_categorical(train_labels, 10)

train_inputs = np.array(train_inputs)#.reshape(len(inputs), -1)

print(train_labels.shape)
#print(inputs.shape)

model = Sequential()

model.add(Flatten())
model.add(Dense(units = 256, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_inputs, train_labels, epochs=10)

test_inputs, test_labels = pickle.load(open('encoder_test.pkl','rb'))

test_inputs = np.array(test_inputs)

test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

pickle.dump(model, open('model_fitted.pkl','wb'))

