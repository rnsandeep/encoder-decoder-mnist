from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.layers import *
from keras.callbacks import TensorBoard
from keras import backend as K
import pickle
from tqdm import tqdm

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = pickle.load(open('encoder.pkl','rb'))

#print(dir(autoencoder))

train_sample = x_train_noisy[:5]

print(train_sample.shape)

encoder_layer = autoencoder.layers[1:][3]
keras_function = K.function([autoencoder.input], [encoder_layer.output])

encoder_train = []
for train_sample in tqdm(x_train_noisy):
    output = keras_function([train_sample.reshape((1, 28, 28, 1)), 1])
    encoder_train.append(output)

pickle.dump([encoder_train, y_train], open('encoder_train.pkl','wb'))


encoder_test = []
for test_sample in tqdm(x_test_noisy):
    output = keras_function([test_sample.reshape((1, 28, 28, 1)), 1])
    encoder_test.append(output)

pickle.dump([encoder_test, y_test], open('encoder_test.pkl','wb'))





