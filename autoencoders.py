# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:14:34 2020

@author: sharm
"""

from keras.layers import Input,Dense
from keras.models import Model

encoding_dim = 32

input_img = Input(shape=(784,))
encoded = Dense(encoding_dim,activation = 'relu')(input_img)
decoded = Dense(784,activation ='sigmoid')(encoded)

autoencoder = Model(input_img,decoded)
print(autoencoder)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]
print(decoder_layer)

decoder = Model(encoded_input, decoder_layer(encoded_input))

print(decoder)

autoencoder.compile(optimizer='adadelta',loss = 'binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(x_train, _),(x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize = (20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

from keras import regularizers
encoding_dim = 32
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim,activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = Dense(784,activation = 'sigmoid')(encoded)
autoencoder = Model(input_img,decoded)


input_img = Input(shape=(784,))
encoded = Dense(128, activation ='relu')(input_img)
encoded = Dense(64, activation ='relu')(encoded)
encoded = Dense(32, activation ='relu')(encoded)

decoded = Dense(64, activation ='relu')(encoded)
decoded = Dense(128, activation ='relu')(decoded)
decoded = Dense(784, activation ='sigmoid')(decoded)

autoencoder =Model(input_img,decoded)
autoencoder.compile(optimizer='adadelta',loss = 'binary_crossentropy')

autoencoder.fit(x_train,x_train,epochs =100,batch_size = 256, shuffle = True, validation_data = (x_test,x_test))

#conv autoencoders

from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(28,28,1))

x = Conv2D(16,(3,3), activation ='relu',padding='same')(input_img)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
